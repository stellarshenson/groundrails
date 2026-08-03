"""R8-H103 - Qwen3-0.6B decoder scorer: the capability-class test.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8;
budget reopened by the author 2026-08-03).

Because Mode C (numeric-derivation blindness) is a pretraining-capability gap
and not an evidence or formula gap, a Qwen3-Reranker-0.6B-initialized
sequence-classification scorer trained one epoch on the identical 762k mix
(BCE, no DANN in stage 1, MAX_LEN 1,024) and read through the frozen gate under
the PRIMARY windowed read will read finqa >= 0.72 while the blind mean lands
>= 0.70 and in-domain gold holds >= 0.80. Kill: instability, finqa < 0.70, or
gold < 0.80 -> the decoder line closes at this size.

Data mix and private trace-split are byte-identical to the holder trainer
(R8-H100_dann_draw3.py, SEED 0); the levers are the model class and its
reranker pair format. Pair format follows the converted checkpoint's own
convention (tomaarsen/Qwen3-Reranker-0.6B-seq-cls): one formatted string per
pair - system prompt + <Instruct>/<Query>/<Document> + assistant suffix -
suffix tokens appended AFTER truncation so the scored final token always
carries the template, left padding, single logit.

Smoke test (200 optimizer steps, separate checkpoint dir, no in-domain eval):
      H103_SMOKE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
      uv run python experiments/grounding-semantic/R8-H103_qwen_scorer.py

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R8-H103_qwen_scorer.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import importlib.util
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
SMOKE = os.environ.get("H103_SMOKE") == "1"
CKPT_DIR = HERE.parent.parent / "models" / ("R8-H103-smoke" if SMOKE else "R8-H103-qwen06b-scorer")
OUT = HERE / ("R8-H103_smoke_result.json" if SMOKE else "R8-H103_result.json")

STUDENT = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"  # 0.6B; sub-400M budget reopened for this line
MAX_LEN = 1024
BATCH = 16  # micro-batch; x ACCUM = the holder's effective 48
ACCUM = 3
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
SEED = 0

# The reranker is instruction-aware; one fixed instruction for the grounding task.
INSTRUCTION = "Given a claim and a document, judge whether the claim is fully supported by the document."
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

BARS = {"gold": (0.7095, 0.76), "ragtruth_en": (0.7039, 0.75), "ragtruth_nonen": (0.6095, 0.66)}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")


def format_pair(claim, chunk):
    """The converted checkpoint's pair convention, minus the suffix (appended as
    token ids after truncation so it survives on every pair)."""
    return f"{PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {claim}\n<Document>: {chunk}"


def encode_pairs(tok, claims, chunks, suffix_ids):
    texts = [format_pair(c, k) for c, k in zip(claims, chunks, strict=True)]
    enc = tok(texts, truncation=True, max_length=MAX_LEN - len(suffix_ids))
    ids = [x + suffix_ids for x in enc["input_ids"]]
    return tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")


def private_train():
    """Soft teacher labels, TRAIN traces only - the SAME seed-0 trace split as the
    substrate's our_gold, so the gold gate stays disjoint. See R8-H81 for why the
    split must not drift."""
    df = pl.read_parquet(PAIRS)
    g = pl.read_parquet(GOLD).with_row_index("owner")
    df = df.join(g.select(["owner", "trace_id"]), on="owner", how="left")
    traces = np.array(sorted(set(df["trace_id"].to_list())))
    rng = np.random.default_rng(SEED)
    rng.shuffle(traces)
    n_test, n_val = int(len(traces) * 0.25), int(len(traces) * 0.15)
    held = set(traces[: n_test + n_val].tolist())
    tr = df.filter(~pl.col("trace_id").is_in(list(held)))
    claims = tr["claim"].to_list()
    chunks = tr["chunk"].to_list()
    y = tr["rerank"].to_numpy().astype("float32")
    return claims, chunks, y


def public_train():
    """The H84 mix (RAGTruth, HaluEval, PsiloQA, VitaminC) + TabFact, RAGBench
    excluded - loading byte-identical to the holder trainer, domain tags dropped
    (no DANN in stage 1)."""
    claims, chunks, ys = [], [], []

    z = zipfile.ZipFile(DATA / "dataset-ragtruth.zip")
    n = next(x for x in z.namelist() if x.endswith("__train.parquet"))
    df = pl.read_parquet(io.BytesIO(z.read(n)))
    df = df.with_columns(
        (
            (pl.col("hallucination_labels_processed").struct.field("evident_conflict") == 0)
            & (pl.col("hallucination_labels_processed").struct.field("baseless_info") == 0)
        )
        .cast(pl.Float32)
        .alias("label")
    ).filter(pl.col("context").str.len_chars() > 50)
    claims += df["output"].to_list()
    chunks += [c[: M59.CFG.chunk_max_chars] for c in df["context"].to_list()]
    ys.append(df["label"].to_numpy())

    zt = zipfile.ZipFile(DATA / "dataset-ragtruth-translated.zip")
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        nm = next(
            x for x in zt.namelist() if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet")
        )
        d = pl.read_parquet(io.BytesIO(zt.read(nm)))
        d = d.with_columns((pl.col("labels").list.len() == 0).cast(pl.Float32).alias("label"))
        d = d.filter(pl.col("prompt").str.len_chars() > 50)
        claims += d["answer"].to_list()
        chunks += [c[: M59.CFG.chunk_max_chars] for c in d["prompt"].to_list()]
        ys.append(d["label"].to_numpy())

    zh = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    for cfg, ev_col, pos_col, neg_col in (
        ("qa", "knowledge", "right_answer", "hallucinated_answer"),
        ("summarization", "document", "right_summary", "hallucinated_summary"),
    ):
        hits = [x for x in zh.namelist() if f"__{cfg}__" in x]
        if not hits:
            continue
        d = pl.read_parquet(io.BytesIO(zh.read(hits[0])))
        if not {ev_col, pos_col, neg_col} <= set(d.columns):
            continue
        for ev, pos, neg in zip(
            d[ev_col].to_list(), d[pos_col].to_list(), d[neg_col].to_list(), strict=True
        ):
            ev = ev[: M59.CFG.chunk_max_chars]
            claims += [pos, neg]
            chunks += [ev, ev]
            ys.append(np.array([1.0, 0.0], dtype="float32"))

    zp = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    dp = pl.read_parquet(
        io.BytesIO(zp.read(next(x for x in zp.namelist() if x.endswith("__train.parquet"))))
    )
    dp = dp.filter(
        (pl.col("wiki_passage").str.len_chars() > 50) & (pl.col("llm_answer").str.len_chars() > 10)
    ).with_columns((pl.col("labels").list.len() == 0).cast(pl.Float32).alias("label"))
    claims += dp["llm_answer"].to_list()
    chunks += [c[: M59.CFG.chunk_max_chars] for c in dp["wiki_passage"].to_list()]
    ys.append(dp["label"].to_numpy())

    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    dv = pl.read_parquet(
        io.BytesIO(zv.read(next(x for x in zv.namelist() if x.endswith("__train.parquet"))))
    )
    lab_col = next(c for c in ("label", "labels") if c in dv.columns)
    ev_col = next(c for c in ("evidence", "wiki_passage", "context") if c in dv.columns)
    cl_col = next(c for c in ("claim", "output", "answer") if c in dv.columns)
    dv = dv.with_columns(
        (pl.col(lab_col).cast(pl.Utf8).str.to_uppercase() == "SUPPORTS")
        .cast(pl.Float32)
        .alias("y")
    )
    claims += dv[cl_col].to_list()
    chunks += [c[: M59.CFG.chunk_max_chars] for c in dv[ev_col].to_list()]
    ys.append(dv["y"].to_numpy())

    zt2 = zipfile.ZipFile(DATA / "dataset-tabfact.zip")
    dt = pl.read_parquet(
        io.BytesIO(zt2.read(next(x for x in zt2.namelist() if x.endswith("__train.parquet"))))
    )
    dt = dt.filter(pl.col("statement").str.len_chars() > 10)
    claims += dt["statement"].to_list()
    chunks += [
        f"{cap}\n{tbl}".replace("\r\n", "\n").replace("#", " | ")[: M59.CFG.chunk_max_chars]
        for cap, tbl in zip(dt["table_caption"].to_list(), dt["table_text"].to_list(), strict=True)
    ]
    ys.append(dt["label"].cast(pl.Float32).to_numpy())

    return claims, chunks, np.concatenate(ys).astype("float32")


class PairSet(Dataset):
    def __init__(self, claims, chunks, y, tok, suffix_ids):
        self.c, self.k, self.y, self.tok, self.suf = claims, chunks, y, tok, suffix_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i]

    def collate(self, b):
        c, k, y = zip(*b, strict=True)
        enc = encode_pairs(self.tok, list(c), list(k), self.suf)
        return enc, torch.tensor(y)


@torch.inference_mode()
def score_pairs(model, tok, suffix_ids, claims, chunk_lists, batch=32):
    """Max-over-chunks over the reranker logit, the cascade serving shape."""
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k)
            owner.append(i)
    out = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), batch):
        enc = encode_pairs(tok, flat_c[i : i + batch], flat_k[i : i + batch], suffix_ids)
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.squeeze(-1)
        out[i : i + batch] = torch.sigmoid(logits.float()).cpu().numpy()
    owner = np.array(owner)
    return np.array([out[owner == i].max() for i in range(len(claims))])


def evaluate(model, tok, suffix_ids):
    """All three corpora, same harness as the recorded bars."""
    res = {}

    claims, chunk_lists, y, _, _ = _mod("sub", "R8_score_substrate.py").our_gold()
    s = score_pairs(model, tok, suffix_ids, claims, chunk_lists)
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["gold"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    cl, ctx, y = M60.load_english()
    s = score_pairs(
        model, tok, suffix_ids, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
    )
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["ragtruth_en"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    per_lang = {}
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        cl, ctx, y = M60.load_translated(lg)
        s = score_pairs(
            model, tok, suffix_ids, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
        )
        auc, f1, _ = M59.auc_and_f1(y, s)
        per_lang[lg] = round(auc, 4)
    res["ragtruth_nonen"] = {
        "auc": round(float(np.mean(list(per_lang.values()))), 4),
        "per_lang": per_lang,
    }
    return res


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    pc, pk, py = private_train()
    uc, uk, uy = public_train()
    claims, chunks = pc + uc, pk + uk
    y = np.concatenate([py, uy])
    print(f"train: {len(y)} pairs, mean target {y.mean():.3f}\n", flush=True)

    tok = AutoTokenizer.from_pretrained(STUDENT, padding_side="left")
    suffix_ids = tok(SUFFIX, add_special_tokens=False)["input_ids"]
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=1, torch_dtype=torch.float32
    ).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {STUDENT}  {n_par:.1f}M params  (budget reopened past 400M)\n", flush=True)

    ds = PairSet(claims, chunks, y, tok, suffix_ids)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=ds.collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_opt_steps = (len(dl) + ACCUM - 1) // ACCUM
    if SMOKE:
        n_opt_steps = 200
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_opt_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    lossf = nn.BCEWithLogitsLoss()

    model.train()
    t0 = time.time()
    opt_step = 0
    for micro, (enc, yy) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        yy = yy.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.squeeze(-1)
            loss = lossf(logits.float(), yy)
        (loss / ACCUM).backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at micro-step {micro}")
        if (micro + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            opt.zero_grad()
            opt_step += 1
            if opt_step % 200 == 0 or opt_step == 1:
                print(
                    f"  step {opt_step}/{n_opt_steps} loss {loss.item():.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if opt_step >= n_opt_steps:
                break

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tok.save_pretrained(CKPT_DIR)
    print(f"\ncheckpoint saved -> {CKPT_DIR}\n", flush=True)

    if SMOKE:
        print("smoke run: skipping in-domain eval", flush=True)
        OUT.write_text(json.dumps({"smoke": True, "steps": opt_step}, indent=2))
        return

    model.eval()
    res = evaluate(model, tok, suffix_ids)
    res["params_M"] = round(n_par, 1)

    print("=" * 96)
    print("R8-H103 RESULT - Qwen3-0.6B decoder scorer, stage 1 (in-domain; blind via R8-H103_read)")
    print("=" * 96)
    print(f"{'corpus':18s} {'ours':>9} {'lettuce':>9} {'delta':>9} {'decisive bar':>13} {'':>6}")
    won = 0
    for key, (bar, decisive) in BARS.items():
        a = res[key]["auc"]
        mark = "DECISIVE" if a >= decisive else ("beat" if a > bar else "LOSE")
        won += a > bar
        print(f"{key:18s} {a:>9.4f} {bar:>9.4f} {a - bar:>+9.4f} {decisive:>13.4f}  {mark}")
    print(f"\n  corpora beaten: {won}/3   params {n_par:.1f}M")
    print("  blind arena: R8-H103_read.py --model models/R8-H103-qwen06b-scorer")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
