"""R4-H29 - POSITIVE CONTROL gate for generative grounding judges.

Methodology rule adopted 2026-07-28 (see the log): validate the instrument BEFORE
the experiment. A judge that cannot separate a claim quoted verbatim from its own
chunk from the same claim against an unrelated recipe cannot be trusted on real
data, and its null tells us nothing.

Gate: the model must (a) emit a parseable verdict on >= 90% of cases and
(b) score >= 9/10 correct on trivially separable pairs. Failing either stops that
candidate before any subsample run is spent.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     uv run python experiments/grounding-semantic/R4-H29_positive_control.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import re  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

FACTS = [
    ("The dam is 271 metres tall.", "The dam is 271 metres tall and produces 1300 MW of power."),
    ("The study reports an RMSE of 7.9 cm.", "The study reports an RMSE of 7.9 cm at 70 m standoff."),
    ("Paris is the capital of France.", "Paris is the capital of France and its largest city."),
    ("Water boils at 100 degrees Celsius.", "Water boils at 100 degrees Celsius at sea level."),
    ("The company employs 4200 people.", "The company employs 4200 people across twelve sites."),
    ("The bridge opened in 1937.", "The bridge opened in 1937 after four years of construction."),
    ("Revenue grew by 12 percent.", "Revenue grew by 12 percent in the third quarter."),
    ("The engine has six cylinders.", "The engine has six cylinders and a turbocharger."),
    ("Crack IoU reached 66.76 percent.", "Crack IoU reached 66.76 percent while background was 99.76."),
    ("The vessel displaces 9000 tonnes.", "The vessel displaces 9000 tonnes when fully loaded."),
]
UNRELATED = (
    "A recipe for sourdough bread requires flour, water, salt and a starter culture. "
    "Prove the dough overnight, then bake at 230 degrees for 40 minutes."
)

# Parser built against OBSERVED output, not an assumed shape. Baguettotron closes
# </think> and then emits a clean "**Answer: Yes, the claim is supported...**" line.
# Parsing the whole generation is WRONG: the reasoning trace is full of phrases like
# "No contradiction. No support needed." and a first-match regex reads that as a NO
# verdict, which scored a correct SUPPORTED case as False. Read the ANSWER, not the
# reasoning.
ANSWER_RE = re.compile(r"answer\s*[:\-]\s*(yes|no)\b", re.I)
YES = re.compile(r"\b(yes|is supported|does support)\b", re.I)
NO = re.compile(r"\b(no|unsupported|not supported|does not support)\b", re.I)


def parse(text):
    """Verdict from the post-</think> answer segment only; None if never committed."""
    seg = text.split("</think>", 1)[1] if "</think>" in text else ""
    if not seg.strip():
        return None  # never closed the trace inside the budget - not a verdict
    m = ANSWER_RE.search(seg)
    if m:
        return m.group(1).lower() == "yes"
    y, n = YES.search(seg), NO.search(seg)
    if y and n:
        return y.start() < n.start()
    if y:
        return True
    if n:
        return False
    return None


def run(model_id, budget=700):
    print(f"\n{'=' * 68}\nCANDIDATE: {model_id}\n{'=' * 68}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    pad = tok.convert_tokens_to_ids("[PAD]")
    cases = [(c, ev, True) for c, ev in FACTS] + [(c, UNRELATED, False) for c, _ in FACTS]
    ok, parsed = 0, 0
    for claim, ev, truth in cases:
        user = (
            f"Evidence:\n{ev}\n\nClaim: {claim}\n\n"
            "Is the claim supported by the evidence? Answer yes or no."
        )
        # the template's add_generation_prompt appends <think>\n - it is part of the PROMPT
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=budget, do_sample=False, pad_token_id=pad
            )
        gen = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
        v = parse(gen)
        parsed += v is not None
        ok += v is truth
        mark = "OK " if v is truth else ("?? " if v is None else "XX ")
        print(f"  {mark}expect={'SUP' if truth else 'UNS'} got={v}  {claim[:44]}", flush=True)
    n = len(cases)
    print(f"\n  parseable verdicts: {parsed}/{n} ({parsed / n:.0%})")
    print(f"  correct on trivial: {ok}/{n} ({ok / n:.0%})")
    passed = parsed / n >= 0.90 and ok / n >= 0.90
    print(f"  GATE: {'PASS - proceed to the subsample run' if passed else 'FAIL - candidate stops here'}")
    del model
    torch.cuda.empty_cache()
    return passed


if __name__ == "__main__":
    for mid in sys.argv[1:] or ["PleIAs/Baguettotron"]:
        run(mid)
