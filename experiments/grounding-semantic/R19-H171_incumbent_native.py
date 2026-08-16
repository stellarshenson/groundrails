"""R19-H171 INCUMBENT UNDER ITS OWN CONVENTION - re-opening a diligence item that was closed wrongly.

WHY THIS EXISTS
---------------
The published headline is "+0.0694 over `KRLabsOrg/lettucedect-v2-mmbert-base` at
0.6461". A diligence item was recorded on 2026-08-15 claiming that margin is
scored "by the INCUMBENT'S OWN convention". Two independent adversarial reviewers
found that claim circular, and they are right: the check re-ran THIS PROJECT's
`R8-H77_unseen_arena.score_lettuce` against THIS PROJECT's banked output and
reported agreement to 1.9e-04. That establishes determinism. It establishes
nothing about whether the convention is the vendor's.

The vendor's actual convention, read from `lettucedetect` 0.2.3 source:

  `preprocess/preprocess_ragbench.py` builds ONE prompt per item -
      context_str = "\\n".join(f"passage {i+1}: {passage}" for i, passage in ...)
      prompt      = PROMPT_QA.format(question=..., num_passages=..., context=context_str)
      answer      = the response
  `detectors/transformer.py` scores it in ONE pass at `max_length=4096`, grouping
  passages into chunks ONLY when the whole template overflows that budget, and
  reads `softmax(logits)[..., 1]` over ANSWER tokens.

What the arena harness does instead (`R8-H77_unseen_arena.score_lettuce`):
  - truncates EACH document to `chunk_max_chars` (1,500)
  - runs each document as a SEPARATE forward pass, with NO question and NO
    instruction template
  - aggregates `max` over documents

Those are different measurements. The vendor gives the model the whole evidence
pool at once with the question; the harness gives it isolated truncated
fragments. This arm measures the incumbent BOTH ways on the IDENTICAL items.

DISCIPLINE
----------
This is not a tuning read and cannot become one. It changes no threshold, selects
no formula, and touches OUR model not at all - our number stays 0.71549 whatever
this returns. It re-measures the BASELINE. There is exactly one honest outcome
rule, fixed here before the run:

  the published margin is recomputed against the incumbent's BEST convention,
  whichever that turns out to be.

Taking the incumbent's worse number because it flatters us is the failure mode
this arm exists to remove.

Item identity is guaranteed by rebuilding the sample through the arena loader's
own filter, `seed=0` sample and `N_PER_SUBSET`, from the same archive member -
so both conventions score the same responses with the same labels.

Run: CUDA_VISIBLE_DEVICES=<gpu> uv run python R19-H171_incumbent_native.py
"""

import importlib.util
import io
import json
import os
import pathlib
import time
import zipfile

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer  # noqa: E402

HERE = pathlib.Path(__file__).parent

# Verbatim from lettucedetect 0.2.3 `preprocess/preprocess_ragbench.py`.
PROMPT_QA = """
Briefly answer the following question:
{question}
Bear in mind that your response should be strictly based on the following {num_passages} passages:
{context}
In case the passages do not contain the necessary information to answer the question, please reply with: "Unable to answer based on given passages."
output:
"""

MAX_LENGTH = 4096          # the vendor's default
BATCH = 4

BANKED_HARNESS = {"covidqa": 0.7355, "delucionqa": 0.7929, "emanual": 0.5999,
                  "expertqa": 0.6503, "finqa": 0.7170, "hagrid": 0.5992,
                  "hotpotqa": 0.5976, "pubmedqa": 0.5162, "tatqa": 0.6156,
                  "techqa": 0.6363}
FLAGSHIP_MEAN = 0.71549


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")
M59 = ARENA.M59


def load_items():
    """The IDENTICAL sample the arena scores, plus the `question` field the
    harness discards. Same archive, same filter, same seed, same N."""
    z = zipfile.ZipFile(ARENA.ARCHIVE)
    out = {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        df = df.sample(min(ARENA.N_PER_SUBSET, len(df)), seed=0)
        out[sub] = (df["question"].to_list(),
                    df["documents"].to_list(),
                    df["response"].to_list(),
                    df["adherence_score"].cast(pl.Int8).to_numpy())
    return out


def build_prompt(question, documents):
    ctx = "\n".join(f"passage {i + 1}: {p}" for i, p in enumerate(documents))
    return PROMPT_QA.format(question=question, num_passages=len(documents), context=ctx)


@torch.inference_mode()
def score_native(tok, model, questions, docs, responses):
    """P(grounded) = 1 - max P(hallucinated) over ANSWER tokens, ONE pass per item
    over the whole passage pool, exactly as the vendor's preprocessing formats it."""
    scores = np.zeros(len(responses), dtype=np.float32)
    over = 0
    for i in range(0, len(responses), BATCH):
        sl = slice(i, i + BATCH)
        prompts = [build_prompt(q, d) for q, d in zip(questions[sl], docs[sl], strict=True)]
        answers = list(responses[sl])
        enc = tok(prompts, answers, truncation="only_first", max_length=MAX_LENGTH,
                  padding=True, return_tensors="pt").to("cuda")
        p = torch.softmax(model(**enc).logits.float(), dim=-1)[..., 1]
        ids = enc["input_ids"]
        sep = tok.sep_token_id
        for r in range(ids.shape[0]):
            row = ids[r].tolist()
            first = row.index(sep) if sep in row else 0
            m = enc["attention_mask"][r].bool().clone()
            m[: first + 1] = False           # answer segment only
            q = p[r][m]
            scores[i + r] = 1.0 - (q.max().item() if q.numel() else 1.0)
        over += int((enc["attention_mask"].sum(1) >= MAX_LENGTH).sum())
    return scores, over


def main():
    out = HERE / "R19-H171_incumbent_native.json"
    print(f"=== R19-H171 incumbent under its OWN convention  {time.strftime('%F %T')} ===",
          flush=True)
    print(f"  device {torch.cuda.get_device_name(0)}  max_length {MAX_LENGTH}", flush=True)

    tok = AutoTokenizer.from_pretrained(ARENA.LETTUCE)
    model = AutoModelForTokenClassification.from_pretrained(
        ARENA.LETTUCE, dtype=torch.float16).cuda().eval()

    items = load_items()
    rows, truncated = {}, {}
    for sub in sorted(items):
        qs, ds, rs, y = items[sub]
        s, over = score_native(tok, model, qs, ds, rs)
        auc, f1, _ = M59.auc_and_f1(np.asarray(y), s)
        rows[sub] = {"n": len(y), "native_auc": round(float(auc), 4),
                     "harness_auc": BANKED_HARNESS[sub],
                     "delta": round(float(auc) - BANKED_HARNESS[sub], 5),
                     "items_at_max_length": int(over)}
        truncated[sub] = int(over)
        print(f"  {sub:12} native {auc:.4f}  harness {BANKED_HARNESS[sub]:.4f}  "
              f"delta {auc - BANKED_HARNESS[sub]:+.4f}  ({over}/{len(y)} hit the cap)",
              flush=True)

    native = float(np.mean([r["native_auc"] for r in rows.values()]))
    harness = float(np.mean([r["harness_auc"] for r in rows.values()]))
    best = max(native, harness)
    res = {
        "arm": "R19-H171 incumbent under its own convention",
        "model": ARENA.LETTUCE,
        "why": ("the recorded diligence item closed on a self-reproduction of this "
                "project's own harness; two adversarial reviewers found that circular"),
        "vendor_convention": {
            "source": "lettucedetect 0.2.3 preprocess_ragbench.py + detectors/transformer.py",
            "prompt": "PROMPT_QA verbatim, question + all passages as 'passage i: ...'",
            "passes": "ONE per item over the whole pool", "max_length": MAX_LENGTH,
            "score": "1 - max P(hallucinated) over answer tokens",
        },
        "harness_convention": {
            "source": "R8-H77_unseen_arena.score_lettuce",
            "prompt": "no question, no template; each document alone",
            "passes": "one per document, each truncated to chunk_max_chars=1500",
            "aggregation": "max over documents",
        },
        "per_subset": rows,
        "native_mean": round(native, 5),
        "harness_mean": round(harness, 5),
        "incumbent_best_mean": round(best, 5),
        "which_is_best": "native" if native > harness else "harness",
        "flagship_mean": FLAGSHIP_MEAN,
        "published_margin_vs_harness": round(FLAGSHIP_MEAN - harness, 5),
        "honest_margin_vs_best": round(FLAGSHIP_MEAN - best, 5),
        "margin_shrinkage": round(best - harness, 5),
        "rule": ("fixed before the run: the published margin is recomputed against "
                 "the incumbent's BEST convention. Taking its worse number because "
                 "that flatters us is the failure this arm exists to remove."),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  incumbent native {native:.5f} | harness {harness:.5f} | "
          f"best {best:.5f}", flush=True)
    print(f"  published margin {FLAGSHIP_MEAN - harness:+.5f}  ->  "
          f"honest margin {FLAGSHIP_MEAN - best:+.5f}  "
          f"(shrinks by {best - harness:.5f})", flush=True)
    print(f"  -> {out.name}", flush=True)
    print("=== H171 COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
