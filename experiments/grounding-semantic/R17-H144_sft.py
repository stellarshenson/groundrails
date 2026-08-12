"""R17-H144 Stage B / Job 2 - student SFT on accepted teacher traces (GPU0).

Full fine-tune of HuggingFaceTB/SmolLM2-360M-Instruct on the teacher traces that
survived the Job 1 rejection filter. The chat turn is the eval turn by
construction:

  user turn      = the Stage-A `chat` prompt (SmolLM2's own ChatML template,
                   system = the banked INSTRUCTION, user = the evidence+claim
                   block) - byte-identical to what the eval harness renders
  assistant turn = "<think>\\n{teacher think}\\n</think>\\n\\nAnswer: {VERDICT}"

`<think>` / `</think>` are ADDED to the tokenizer as special tokens (the trlm
recipe - the delimiters are load-bearing, and SmolLM2 ships neither). Loss is on
assistant tokens only; the prompt is masked with -100.

Precision: parameters are held in fp32 with bf16 autocast rather than cast to
bf16 outright - at lr 3e-5 a bf16 parameter cannot represent the update (the
step lands below one ulp) and training stalls silently. The compute is bf16.

Sequence budget 1024. Where prompt + trace overflows it, EVIDENCE rows are
dropped - never the think or the verdict - keeping the table title, the markdown
header and every row naming the claim's own keys; a row that still overflows is
dropped and counted.

Held-out: 5% of DOCUMENTS (not rows), so no validation claim shares a table with
a training claim.

Guard: after epoch 1, a validation verdict-format rate below 0.9 stops the run.

Run (detached):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
  nohup setsid python experiments/grounding-semantic/R17-H144_sft.py \\
      >> logs/R17-H144_sft.log 2>&1 &
"""

import importlib.util
import json
import math
import os
import pathlib
import random
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# NOT expandable_segments - WSL2's driver rejects the VMM calls it needs and the
# model .to("cuda") dies with "CUDA driver error: unknown error" (measured)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
PAIRS = HERE / "R17-H144_pairs.parquet"
TRACES = HERE / "R17-H144_traces.parquet"
LOOKUP = HERE / "R17-H144_lookup.parquet"
# amendment A1: cycle 2 is a FRESH fine-tune from the base on the mixed corpus
# (teacher traces + the verbatim-lookup family), never a continuation of cycle
# 1's epoch 3 - the comparison has to be recipe-identical and init-identical
CYCLE2 = os.environ.get("H144_CYCLE2") == "1"
OUTDIR = ROOT / "models" / ("R17-H144-student-c2" if CYCLE2 else "R17-H144-student")
STATS = HERE / ("R17-H144_sft_c2_stats.json" if CYCLE2 else "R17-H144_sft_stats.json")

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_LEN = 1024
EPOCHS = 3
LR = 3e-5
MICRO_BS = 8
ACCUM = 8
WARMUP_FRAC = 0.03
VAL_DOC_FRAC = 0.05
FORMAT_PROBE_N = 200
FORMAT_BAR = 0.90
SEED = 1144

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _stage_a():
    spec = importlib.util.spec_from_file_location("sa", HERE / "R17-H143_stageA.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _stage_a()


# --------------------------------------------------------------------------- #
# evidence trimming - drop evidence rows, never the think or the verdict
# --------------------------------------------------------------------------- #
def _units(chunk: str) -> tuple[list[str], str]:
    if "\n" in chunk:
        return chunk.split("\n"), "\n"
    # FEVEROUS-style serialisation: one line of sentences
    parts = [p for p in chunk.split(". ") if p]
    return [p if p.endswith(".") else p + "." for p in parts], " "


def trim_chunk(chunk: str, keys: list[str], n_over: int, tok) -> str | None:
    """Drop droppable evidence units until the example fits, or give up."""
    units, join = _units(chunk)
    protected = {0}
    if join == "\n" and len(units) > 2 and set(units[2].replace("|", "").strip()) <= {"-", " "}:
        protected |= {1, 2}
    low = [u.lower() for u in units]
    for i, u in enumerate(low):
        if any(k and k.lower() in u for k in keys):
            protected.add(i)
    droppable = [i for i in range(len(units)) if i not in protected]
    dropped: set[int] = set()
    for i in reversed(droppable):
        dropped.add(i)
        cut = sum(len(tok.encode(units[j] + join, add_special_tokens=False)) for j in dropped)
        if cut >= n_over:
            return join.join(u for j, u in enumerate(units) if j not in dropped)
    if not droppable:
        return None
    left = join.join(u for j, u in enumerate(units) if j not in dropped)
    return left if len(tok.encode(left, add_special_tokens=False)) else None


# --------------------------------------------------------------------------- #
def build_examples(tok) -> tuple[list[dict], dict]:
    tr = pl.read_parquet(TRACES).filter(pl.col("accepted"))
    pairs = pl.read_parquet(PAIRS).with_columns(pl.col("label").cast(pl.Int64))
    d = tr.select(["pair_id", "label", "think", "verdict", "neg_family"]).join(
        pairs.select(["pair_id", "label", "claim", "chunk", "doc_id", "key_a", "key_b"]),
        on=["pair_id", "label"], how="inner",
    )
    n_deriv = d.height
    n_lookup = 0
    if CYCLE2:
        lk = pl.read_parquet(LOOKUP).with_columns(
            (pl.col("pair_id") + 10_000_000).alias("pair_id")).select(d.columns)
        n_lookup = lk.height
        d = pl.concat([d, lk], how="vertical_relaxed")
    print(f"[data] accepted traces {tr.height}, joined {n_deriv}, "
          f"lookup {n_lookup}, corpus {d.height}", flush=True)

    eos = tok.eos_token or "<|im_end|>"
    out, n_trim, n_drop = [], 0, 0
    for r in d.to_dicts():
        # the teacher trace already opens with its own <think> - the target must
        # carry exactly one opening delimiter, not two
        think = (r["think"] or "").strip()
        if think.startswith(THINK_OPEN):
            think = think[len(THINK_OPEN):].strip()
        target = f"{THINK_OPEN}\n{think}\n{THINK_CLOSE}\n\nAnswer: {r['verdict']}{eos}"
        t_ids = tok.encode(target, add_special_tokens=False)
        chunk = r["chunk"]
        p_ids = tok.encode(SA.build_prompt("chat", tok, chunk, r["claim"]),
                           add_special_tokens=False)
        if len(p_ids) + len(t_ids) > MAX_LEN:
            new = trim_chunk(chunk, [r["key_a"], r["key_b"]],
                             len(p_ids) + len(t_ids) - MAX_LEN, tok)
            if new is None:
                n_drop += 1
                continue
            chunk = new
            p_ids = tok.encode(SA.build_prompt("chat", tok, chunk, r["claim"]),
                               add_special_tokens=False)
            n_trim += 1
            if len(p_ids) + len(t_ids) > MAX_LEN:
                n_drop += 1
                n_trim -= 1
                continue
        out.append(dict(ids=p_ids + t_ids, n_prompt=len(p_ids), doc_id=r["doc_id"],
                        label=r["label"], neg_family=r["neg_family"],
                        chunk=chunk, claim=r["claim"], verdict=r["verdict"]))
    stats = dict(n_accepted=tr.height, n_derivation=n_deriv, n_lookup=n_lookup,
                 cycle2=CYCLE2, n_examples=len(out), n_evidence_trimmed=n_trim,
                 n_dropped_overlong=n_drop,
                 mean_len=float(np.mean([len(e["ids"]) for e in out])))
    print(f"[data] {stats}", flush=True)
    return out, stats


def split_by_doc(exs: list[dict]) -> tuple[list[dict], list[dict]]:
    docs = sorted({e["doc_id"] for e in exs})
    rng = random.Random(SEED)
    rng.shuffle(docs)
    n_val = max(1, int(round(len(docs) * VAL_DOC_FRAC)))
    val_docs = set(docs[:n_val])
    tr = [e for e in exs if e["doc_id"] not in val_docs]
    va = [e for e in exs if e["doc_id"] in val_docs]
    print(f"[data] {len(docs)} docs -> train {len(tr)} rows / val {len(va)} rows "
          f"({n_val} val docs)", flush=True)
    return tr, va


def collate(batch: list[dict], pad_id: int, device):
    n = max(len(e["ids"]) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, e in enumerate(batch):
        L = len(e["ids"])
        ids[i, :L] = torch.tensor(e["ids"])
        att[i, :L] = 1
        lab[i, e["n_prompt"]:L] = torch.tensor(e["ids"][e["n_prompt"]:])
    return ids.to(device), att.to(device), lab.to(device)


@torch.no_grad()
def val_loss(model, val: list[dict], pad_id: int, device) -> float:
    model.eval()
    tot, ntok = 0.0, 0
    for s in range(0, len(val), MICRO_BS):
        ids, att, lab = collate(val[s:s + MICRO_BS], pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits
        st = lab[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(), st.reshape(-1),
            ignore_index=-100, reduction="sum")
        tot += float(loss)
        ntok += int((st != -100).sum())
    model.train()
    return tot / max(ntok, 1)


@torch.no_grad()
def format_probe(model, tok, val: list[dict], device) -> dict:
    """Greedy-generate on a val slice; how often is the trained format produced?"""
    model.eval()
    rng = random.Random(SEED)
    sub = val if len(val) <= FORMAT_PROBE_N else rng.sample(val, FORMAT_PROBE_N)
    ok = agree = 0
    tok.padding_side = "left"
    for s in range(0, len(sub), 16):
        part = sub[s:s + 16]
        prompts = [SA.build_prompt("chat", tok, e["chunk"], e["claim"]) for e in part]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**enc, max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for e, row in zip(part, new, strict=True):
            text = tok.decode(row, skip_special_tokens=False)
            v, _ = SA.parse_verdict(text.split(THINK_CLOSE)[-1], first=True)
            good = THINK_CLOSE in text and v is not None
            ok += good
            agree += good and v == e["verdict"]
    model.train()
    return dict(n=len(sub), format_rate=ok / len(sub), verdict_agree=agree / len(sub))


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    device = "cuda"
    torch.manual_seed(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    added = tok.add_special_tokens(
        {"additional_special_tokens": [THINK_OPEN, THINK_CLOSE]})
    assert len(tok.encode(THINK_OPEN, add_special_tokens=False)) == 1
    assert len(tok.encode(THINK_CLOSE, add_special_tokens=False)) == 1
    print(f"[tok] added {added} special tokens, vocab now {len(tok)}", flush=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    exs, dstats = build_examples(tok)
    train, val = split_by_doc(exs)

    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    # gradient checkpointing is a SPEED fix here, not only a memory one: without
    # it the step peaks at 22.8 GiB on a 24 GiB card and the caching allocator
    # thrashes - measured 3.89 s/micro-step against 0.755 s/micro-step with it
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    steps_per_epoch = math.ceil(len(train) / (MICRO_BS * ACCUM))
    total_steps = steps_per_epoch * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * WARMUP_FRAC), total_steps)
    print(f"[train] {len(train)} rows, {steps_per_epoch} opt-steps/epoch, "
          f"{total_steps} total", flush=True)

    rng = random.Random(SEED)
    hist: list[dict] = []
    t0 = time.time()
    stopped = None
    for ep in range(1, EPOCHS + 1):
        order = list(train)
        rng.shuffle(order)
        run_loss, run_n, step = 0.0, 0, 0
        ep_loss, ep_n = 0.0, 0
        for s in range(0, len(order), MICRO_BS * ACCUM):
            # length-sorted WITHIN the accumulation group only: the group itself
            # is a random draw, so this costs no shuffling and cuts pad tokens
            group = sorted(order[s:s + MICRO_BS * ACCUM], key=lambda e: len(e["ids"]))
            opt.zero_grad(set_to_none=True)
            for m0 in range(0, len(group), MICRO_BS):
                ids, att, lab = collate(group[m0:m0 + MICRO_BS], tok.pad_token_id, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=att).logits
                st = lab[:, 1:]
                # flat CE with ignore_index, NOT a boolean-mask gather: masking
                # [B,T,V] logits allocates a second copy of the whole tensor and
                # syncs on nonzero() - measured 10x slower end to end
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    st.reshape(-1), ignore_index=-100)
                (loss * len(group[m0:m0 + MICRO_BS]) / len(group)).backward()
                nt = int((st != -100).sum())
                lv = float(loss.detach())
                run_loss += lv * nt
                run_n += nt
                ep_loss += lv * nt
                ep_n += nt
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[ep{ep}] step {step}/{steps_per_epoch} "
                      f"loss {run_loss / max(run_n, 1):.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                run_loss, run_n = 0.0, 0
            if not math.isfinite(lv):
                stopped = f"non-finite loss at epoch {ep} step {step}"
                break
        if stopped:
            break

        vl = val_loss(model, val, tok.pad_token_id, device)
        model.config.use_cache = True
        fp = format_probe(model, tok, val, device)
        model.config.use_cache = False
        hist.append(dict(epoch=ep, train_loss=ep_loss / max(ep_n, 1),
                         train_loss_tail=run_loss / run_n if run_n else None,
                         val_loss=vl, **fp))
        print(f"[ep{ep}] val_loss {vl:.4f} format {fp['format_rate']:.3f} "
              f"agree {fp['verdict_agree']:.3f}", flush=True)

        ckpt = OUTDIR / f"epoch{ep}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        STATS.write_text(json.dumps(dict(data=dstats, history=hist, stopped=stopped,
                                         config=dict(epochs=EPOCHS, lr=LR,
                                                     micro_bs=MICRO_BS, accum=ACCUM,
                                                     max_len=MAX_LEN, seed=SEED)),
                                    indent=2))
        if ep == 1 and fp["format_rate"] < FORMAT_BAR:
            stopped = (f"format_rate {fp['format_rate']:.3f} < {FORMAT_BAR} after "
                       f"epoch 1 - stopping rather than burning epochs")
            print(f"[guard] {stopped}", flush=True)
            break

    best = min(hist, key=lambda h: h["val_loss"]) if hist else None
    STATS.write_text(json.dumps(dict(data=dstats, history=hist, stopped=stopped,
                                     best_epoch=best["epoch"] if best else None,
                                     config=dict(epochs=EPOCHS, lr=LR, micro_bs=MICRO_BS,
                                                 accum=ACCUM, max_len=MAX_LEN, seed=SEED)),
                                indent=2))
    print(f"[sft] DONE stopped={stopped} best_epoch={best['epoch'] if best else None}",
          flush=True)


if __name__ == "__main__":
    main()
