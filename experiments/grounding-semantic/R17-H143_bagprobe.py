"""R17-H143 Stage A - Baguettotron control-gate diagnostic.

Baguettotron fails the positive-control gate twice (0.40 free-parse, 0.366
forced-answer) by answering GROUNDED to every trivially separable control and
UNGROUNDED to none. The gate exists to separate a harness defect from model
behaviour, so this probe varies the two things the harness controls - the think
budget and the instruction phrasing - and reads the verdict margin directly.

  think budget 0    the block is closed immediately: verdict with no reasoning
  think budget 128  the trace is cut early, before it degenerates
  (512 is the main run)

  phrasing A  registered wording
  phrasing B  option order reversed - tests answer-order / recency bias
  phrasing C  explicit decision rule for each verdict

If no cell discriminates on trivial pairs, the constant-GROUNDED read is the
model, not the harness.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
     python experiments/grounding-semantic/R17-H143_bagprobe.py
"""

import json
import os
import pathlib

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "stagea", pathlib.Path(__file__).parent / "R17-H143_stageA.py"
)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R17-H143_bagprobe.json"
MODEL = "PleIAs/Baguettotron"

PHRASINGS = {
    "A_registered": S.INSTRUCTION,
    "B_reversed_order": (
        "You verify whether a claim is grounded in the evidence. The claim may state a "
        "value derived from numbers in the evidence. Check the arithmetic. Answer with "
        "exactly one word: UNGROUNDED or GROUNDED."
    ),
    "C_explicit_rule": (
        "You verify whether a claim is grounded in the evidence. If every value in the "
        "claim is stated in the evidence or correctly computed from it, answer GROUNDED. "
        "If any value is wrong, missing, or miscomputed, answer UNGROUNDED. Check the "
        "arithmetic. Answer with exactly one word."
    ),
}
BUDGETS = [0, 128]


def main() -> None:
    ev = pl.read_parquet(HERE / "R17-H143_evalset.parquet")
    ctrl = ev.filter(pl.col("control")).to_dicts()
    labels = np.array([r["label"] for r in ctrl])

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = "[PAD]"
    g_ids, u_ids = S.verdict_token_ids(tok)
    gt = torch.tensor(g_ids, device="cuda:0")
    ut = torch.tensor(u_ids, device="cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    eos = [tok.get_vocab()[t] for t in ("<|im_end|>", "<|end_of_text|>") if t in tok.get_vocab()]

    def margin_at_answer(prompts: list[str]) -> np.ndarray:
        """Log-prob margin on the first answer token, one forward pass."""
        out = []
        for i in range(0, len(prompts), 25):
            enc = tok(
                prompts[i : i + 25], return_tensors="pt", padding=True, add_special_tokens=False
            ).to("cuda:0")
            with torch.no_grad():
                lg = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(lg, dim=-1)
            out.append(
                (torch.logsumexp(lp[:, gt], -1) - torch.logsumexp(lp[:, ut], -1)).cpu().numpy()
            )
        return np.concatenate(out)

    results = {}
    for pname, instr in PHRASINGS.items():
        for budget in BUDGETS:
            base = [
                tok.apply_chat_template(
                    [{"role": "user",
                      "content": f"{instr}\n\n{S.user_block(r['chunk'], r['claim'])}"}],
                    tokenize=False, add_generation_prompt=True,
                )
                for r in ctrl
            ]
            if budget == 0:
                fp = [p + "\n</think>\nAnswer:" for p in base]
            else:
                fp = []
                for i in range(0, len(base), 25):
                    enc = tok(
                        base[i : i + 25], return_tensors="pt", padding=True,
                        add_special_tokens=False,
                    ).to("cuda:0")
                    with torch.no_grad():
                        gen = model.generate(
                            **enc, max_new_tokens=budget, do_sample=False,
                            eos_token_id=eos or None, pad_token_id=tok.pad_token_id,
                        )
                    new = gen[:, enc["input_ids"].shape[1] :]
                    for j in range(new.shape[0]):
                        t = tok.decode(new[j], skip_special_tokens=True).split("</think>")[0]
                        fp.append(base[i + j] + t + "\n</think>\nAnswer:")
            m = margin_at_answer(fp)
            au = S.auroc(labels, m)
            said_g = int((m > 0).sum())
            results[f"{pname}|think{budget}"] = dict(
                control_auroc=au, n=len(m), said_grounded=said_g,
                said_ungrounded=int(len(m) - said_g),
                mean_margin_pos=float(m[labels == 1].mean()),
                mean_margin_neg=float(m[labels == 0].mean()),
            )
            print(f"{pname:18s} think={budget:3d} AUROC={au:.4f} "
                  f"G={said_g}/{len(m)} mpos={m[labels == 1].mean():+.2f} "
                  f"mneg={m[labels == 0].mean():+.2f}", flush=True)

    OUT.write_text(json.dumps(results, indent=2))
    best = max(results.items(), key=lambda kv: kv[1]["control_auroc"] or 0)
    print(f"\nbest cell: {best[0]} -> {best[1]['control_auroc']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
