"""R19-H171b INCUMBENT, FULL VENDOR PATH - removing the lower-bound caveat.

`R19-H171_incumbent_native.py` scored the incumbent under its own prompt format
but TRUNCATED any prompt over 4,096 tokens, where the vendor instead GROUPS the
passages into chunks and aggregates across them. 193/250 techqa items and 30/203
expertqa items hit that cap, so those two subsets were recorded as lower bounds.

This implements the vendor path exactly, from `lettucedetect` 0.2.3
`detectors/transformer.py`:

  `_group_passages_into_chunks`
      answer_tokens  = tokens(answer, no specials)
      total_budget   = max_length - answer_tokens - 3
      if tokens(full prompt) <= total_budget            -> ONE group
      instruction_overhead = tokens(template with [""])
      passage_budget = total_budget - instruction_overhead
      greedy fill: each passage line "passage i: <text>", +1 token per newline
                   when the group is non-empty; overflow starts a new group

  `_predict_chunked`
      per ANSWER TOKEN, P(hallucinated) = MAX across chunks
      "conservative: a token is only considered supported if EVERY chunk
       considers it supported"

Our response-level score stays `1 - max P(hallucinated) over answer tokens`, so
with the vendor's cross-chunk max this equals
`1 - max over (chunk, answer token)`.

WHICH DIRECTION THIS MOVES IS NOT ASSUMED. Truncation drops passages, which
should depress support. Chunking keeps every passage but scores a token as
hallucinated if ANY chunk fails to support it, which is stricter. The two push
opposite ways and the net is measured here rather than guessed - the earlier
block's "probably higher" was a guess and is retracted by this measurement
whichever way it lands.

Run: CUDA_VISIBLE_DEVICES=<gpu> uv run python R19-H171_incumbent_chunked.py
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

PROMPT_QA = """
Briefly answer the following question:
{question}
Bear in mind that your response should be strictly based on the following {num_passages} passages:
{context}
In case the passages do not contain the necessary information to answer the question, please reply with: "Unable to answer based on given passages."
output:
"""

MAX_LENGTH = 4096
HARNESS = {"covidqa": 0.7355, "delucionqa": 0.7929, "emanual": 0.5999,
           "expertqa": 0.6503, "finqa": 0.7170, "hagrid": 0.5992,
           "hotpotqa": 0.5976, "pubmedqa": 0.5162, "tatqa": 0.6156,
           "techqa": 0.6363}
NATIVE_TRUNC = {"covidqa": 0.7432, "delucionqa": 0.7018, "emanual": 0.7694,
                "expertqa": 0.8098, "finqa": 0.6137, "hagrid": 0.7542,
                "hotpotqa": 0.6161, "pubmedqa": 0.6070, "tatqa": 0.5275,
                "techqa": 0.6536}
FLAGSHIP = 0.71549


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ARENA = _mod("arena", "R8-H77_unseen_arena.py")
M59 = ARENA.M59


def load_items():
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
        out[sub] = (df["question"].to_list(), df["documents"].to_list(),
                    df["response"].to_list(),
                    df["adherence_score"].cast(pl.Int8).to_numpy())
    return out


def fmt(passages, question):
    ctx = "\n".join(f"passage {i + 1}: {p}" for i, p in enumerate(passages))
    return PROMPT_QA.format(question=question, num_passages=len(passages), context=ctx)


def ntok(tok, s):
    return tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]


def group_passages(tok, passages, question, answer):
    """`_group_passages_into_chunks`, transcribed."""
    total_budget = MAX_LENGTH - ntok(tok, answer) - 3
    if ntok(tok, fmt(passages, question)) <= total_budget:
        return [list(passages)]
    overhead = ntok(tok, fmt([""], question))
    passage_budget = total_budget - overhead
    if passage_budget <= 0:
        return [list(passages)]
    counts = [ntok(tok, f"passage {i + 1}: {p}") for i, p in enumerate(passages)]
    groups, cur, cur_tok = [], [], 0
    for p, c in zip(passages, counts, strict=True):
        eff = c + (1 if cur else 0)
        if cur_tok + eff > passage_budget and cur:
            groups.append(cur)
            cur, cur_tok, eff = [], 0, c
        cur.append(p)
        cur_tok += eff
    if cur:
        groups.append(cur)
    return groups or [list(passages)]


@torch.inference_mode()
def answer_token_probs(tok, model, prompt, answer):
    enc = tok(prompt, answer, truncation="only_first", max_length=MAX_LENGTH,
              return_tensors="pt").to("cuda")
    p = torch.softmax(model(**enc).logits.float(), dim=-1)[0, :, 1]
    row = enc["input_ids"][0].tolist()
    sep = tok.sep_token_id
    first = row.index(sep) if sep in row else 0
    m = enc["attention_mask"][0].bool().clone()
    m[: first + 1] = False
    return p[m]


def score_item(tok, model, question, passages, answer):
    groups = group_passages(tok, passages, question, answer)
    per_chunk = [answer_token_probs(tok, model, fmt(g, question), answer) for g in groups]
    n = min(t.numel() for t in per_chunk)
    if n == 0:
        return 0.0, len(groups)
    stacked = torch.stack([t[:n] for t in per_chunk])   # [chunks, tokens]
    agg = stacked.max(dim=0).values                     # vendor: max across chunks
    return 1.0 - float(agg.max()), len(groups)


def main():
    out = HERE / "R19-H171_incumbent_chunked.json"
    print(f"=== R19-H171b incumbent, FULL vendor path  {time.strftime('%F %T')} ===",
          flush=True)
    tok = AutoTokenizer.from_pretrained(ARENA.LETTUCE)
    model = AutoModelForTokenClassification.from_pretrained(
        ARENA.LETTUCE, dtype=torch.float16).cuda().eval()

    rows = {}
    for sub, (qs, ds, rs, y) in sorted(load_items().items()):
        s = np.zeros(len(rs), dtype=np.float32)
        multi = 0
        for i, (q, d, r) in enumerate(zip(qs, ds, rs, strict=True)):
            s[i], g = score_item(tok, model, q, d, r)
            multi += g > 1
        auc, _f1, _ = M59.auc_and_f1(np.asarray(y), s)
        rows[sub] = {"n": len(y), "chunked_auc": round(float(auc), 4),
                     "native_truncated_auc": NATIVE_TRUNC[sub],
                     "harness_auc": HARNESS[sub],
                     "items_needing_multiple_chunks": int(multi),
                     "delta_vs_harness": round(float(auc) - HARNESS[sub], 5)}
        print(f"  {sub:12} chunked {auc:.4f}  trunc {NATIVE_TRUNC[sub]:.4f}  "
              f"harness {HARNESS[sub]:.4f}  ({multi}/{len(y)} multi-chunk)", flush=True)

    ch = float(np.mean([r["chunked_auc"] for r in rows.values()]))
    nt = float(np.mean(list(NATIVE_TRUNC.values())))
    hm = float(np.mean(list(HARNESS.values())))
    best = max(ch, nt, hm)
    which = {"chunked": ch, "native_truncated": nt, "harness": hm}
    res = {"arm": "R19-H171b incumbent, full vendor chunking path",
           "model": ARENA.LETTUCE, "max_length": MAX_LENGTH,
           "per_subset": rows,
           "means": {k: round(v, 5) for k, v in which.items()},
           "incumbent_best_mean": round(best, 5),
           "which_is_best": max(which, key=which.get),
           "flagship_mean": FLAGSHIP,
           "published_margin_vs_harness": round(FLAGSHIP - hm, 5),
           "honest_margin_vs_best": round(FLAGSHIP - best, 5),
           "margin_shrinkage": round(best - hm, 5),
           "rule": "the margin is recomputed against the incumbent's BEST convention",
           "note": "Numbers recorded, not adjudicated - the coordinator adjudicates."}
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  chunked {ch:.5f} | truncated {nt:.5f} | harness {hm:.5f}", flush=True)
    print(f"  incumbent BEST {best:.5f} ({max(which, key=which.get)})", flush=True)
    print(f"  published margin {FLAGSHIP - hm:+.5f} -> honest {FLAGSHIP - best:+.5f}",
          flush=True)
    print("=== H171b COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
