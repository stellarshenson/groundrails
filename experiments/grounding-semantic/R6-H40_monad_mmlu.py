"""R6-H40 - reproduce Monad's published benchmark before judging its format.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 6).

The question this settles: was R4-H29's Monad prompt ever valid?

Monad's `chat_template.json` emits `<|im_start|>`, `<|im_end|>` and `<think>`,
none of which are tokens in its 8,192-slot vocabulary - they shatter into 3-5
BPE pieces. That looks like a defect. But Monad's `added_tokens` list holds only
[UNK]/<|begin_of_text|>/<|end_of_text|>/[PAD], its BPE fills every embedding
slot, and its model card still states it was trained on "the standard
instruction style from Qwen" with that exact block. If PleIAs trained THROUGH
this tokenizer, the model saw the same shattered pieces in training and the
round-4 prompt was faithful.

The discriminator is the published number, not an argument. The card reports
MMLU "close to 30% of positive rate" against a 25% chance floor. PleIAs
published no eval code for Monad, so rather than guess one protocol this
brackets the three standard ones and reports all:

  loglik-0shot  harness-style: logP of each choice's letter, no examples
  loglik-5shot  harness-style with five dev examples, the Open-LLM-Leaderboard shape
  chatml-gen    the card's own instruction block, answer parsed from the trace

A configuration that clears chance by the published margin proves the format it
used. If NONE does, the format question is unresolved and no Monad number in
this repo can be trusted either way - which is itself the finding.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R6-H40_monad_mmlu.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import json
import pathlib
import re

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "PleIAs/Monad"
OUT = pathlib.Path(__file__).parent / "R6-H40_mmlu.json"
N_QUESTIONS = 2000  # se ~1.0pp at p=0.3 - enough to separate 30% from the 25% floor
LETTERS = ["A", "B", "C", "D"]
PUBLISHED = 0.30  # model card: MMLU "close to 30% of positive rate"
CHANCE = 0.25


def render(q, choices, with_answer=None):
    body = f"{q.strip()}\n" + "".join(f"{LETTERS[i]}. {c}\n" for i, c in enumerate(choices))
    tail = f"Answer: {LETTERS[with_answer]}\n\n" if with_answer is not None else "Answer:"
    return body + tail


@torch.inference_mode()
def loglik_eval(model, tok, rows, shots):
    """Harness-style: score logP of ' A'..' D' at the answer position."""
    ids = [tok.encode(f" {ltr}", add_special_tokens=False)[0] for ltr in LETTERS]
    prefix = "".join(render(r["question"], r["choices"], r["answer"]) for r in shots)
    correct = 0
    for i, r in enumerate(rows):
        prompt = prefix + render(r["question"], r["choices"])
        enc = tok(prompt, return_tensors="pt").to("cuda")
        logits = model(**enc).logits[0, -1].float()
        pred = int(torch.tensor([logits[t] for t in ids]).argmax())
        correct += pred == r["answer"]
        if i % 500 == 0:
            print(f"    {i}/{len(rows)}  running {correct / max(i, 1):.3f}", flush=True)
    return correct / len(rows)


@torch.inference_mode()
def chatml_eval(model, tok, rows, max_new=200):
    """The card's own instruction block. The <think> tag is part of the PROMPT."""
    pad = tok.convert_tokens_to_ids("[PAD]")
    ans_re = re.compile(r"\b([ABCD])\b")
    correct = parsed = 0
    for i, r in enumerate(rows):
        user = render(r["question"], r["choices"]).replace(
            "Answer:", "Answer with a single letter: A, B, C or D."
        )
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        enc = tok(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
        gen = tok.decode(out[0][enc.input_ids.shape[1] :], skip_special_tokens=False)
        tail = gen.split("</think>")[-1] if "</think>" in gen else gen
        m = ans_re.search(tail)
        if m:
            parsed += 1
            correct += LETTERS.index(m.group(1)) == r["answer"]
        if i % 100 == 0:
            print(f"    {i}/{len(rows)}  parsed {parsed} correct {correct}", flush=True)
    return correct / len(rows), parsed / len(rows)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).cuda().eval()
    print(
        f"loaded {MID}  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True
    )

    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    rows = list(test.shuffle(seed=0).select(range(min(N_QUESTIONS, len(test)))))
    shots = list(dev.select(range(5)))
    print(
        f"MMLU: {len(rows)} test questions, {len(shots)} dev shots, chance {CHANCE:.0%}\n",
        flush=True,
    )

    results = {}
    print("  [1/3] loglik-0shot", flush=True)
    results["loglik_0shot"] = loglik_eval(model, tok, rows, [])
    print("  [2/3] loglik-5shot", flush=True)
    results["loglik_5shot"] = loglik_eval(model, tok, rows, shots)
    print("  [3/3] chatml-gen (slow - 300 questions)", flush=True)
    acc, parse_rate = chatml_eval(model, tok, rows[:300])
    results["chatml_gen"] = acc
    results["chatml_parse_rate"] = parse_rate

    print("\n" + "=" * 90)
    print("R6-H40 RESULT - reproduce Monad's published MMLU")
    print("=" * 90)
    print(f"  published (model card)   : ~{PUBLISHED:.0%}   chance floor {CHANCE:.0%}")
    for k in ("loglik_0shot", "loglik_5shot", "chatml_gen"):
        v = results[k]
        verdict = "REPRODUCES" if v >= 0.28 else ("above chance" if v >= 0.27 else "at chance")
        print(f"  {k:22s}   : {v:.3f}   ({v - CHANCE:+.3f} vs chance)   {verdict}")
    print(f"  chatml parse rate        : {results['chatml_parse_rate']:.0%}")
    best = max(results[k] for k in ("loglik_0shot", "loglik_5shot", "chatml_gen"))
    print(f"\n  best {best:.3f}")
    if best >= 0.28:
        print("  -> a standard protocol reproduces the published number; Monad's trained")
        print("     format is usable and the R4-H29 Monad row is NOT void on format grounds")
    else:
        print("  -> no standard protocol reaches the published number; the harness question")
        print("     is unresolved and no Monad grounding number here can be trusted")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
