"""R8-H73 - two heads on one trunk: score regression + token-span tagging.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8),
re-stated against the post-H62 bars. Runs LAST in the arm per the ordering.

The motivating fact is R8-H64: our score-regression student and the incumbent's
token-span tagger are ORTHOGONAL (Spearman -0.046..+0.083 on every corpus), and
untrained rank-average fusion beat both. Same evidence, two supervision shapes,
near-zero correlation - so the ensemble headroom is real but costs 875M params
across two models. This hypothesis puts both signals on ONE 307M trunk:

  head A (score)  - Linear([CLS]) -> 1 logit, BCE against the same mixed
                    soft-teacher / hard-human labels every incarnation used;
                    trained on ALL pairs
  head B (tokens) - Linear(hidden) -> 2 logits per token, CE over the CLAIM
                    tokens only, trained where span supervision exists:
                      ragtruth_en          JSON span annotations on the answer
                      ragtruth_<lang>      {start,end,label} char spans (MT-carried)
                      psiloqa              [start,end] char span pairs
                      halueval             whole-answer spans (right=all 0 /
                                           hallucinated=all 1)
                    private (no spans) and vitaminc (atomic claims - a REFUTES
                    is a wrong claim, not a hallucinated span) are masked out

Inference fuses the heads per pair - p = (sigmoid(score) + (1 - max halluc-token
prob)) / 2 - then max-over-chunks per claim, exactly the serving shape. Per-head
AUCs are also recorded, so whether the two heads stayed orthogonal INSIDE one
trunk is measured, not assumed.

Data mix and recipe are byte-identical to R8-H84 (the best blind mean, 0.6450);
the only lever is the second head and its loss term. Registered bars: beat
0.8531 / 0.8434 / 0.8407 in-domain AND blind mean >= 0.68 with all three
holding; the honest comparison line on the arena is H84's 0.6450 and the
incumbent's 0.6461.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R8-H73_twohead.py
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
from transformers import AutoModel, AutoTokenizer

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
CKPT_DIR = HERE.parent.parent / "models" / "R8-H73-mmbert-twohead"
OUT = HERE / "R8-H73_result.json"

STUDENT = "jhu-clsp/mmBERT-base"
MAX_LEN = 512
BATCH = 32
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
W_TOK = 1.0  # weight of the token-head loss term
N_PRIVATE = 40_000
N_PUBLIC_EN = 15_000
N_PUBLIC_PER_LANG = 4_000
N_HALUEVAL = 6_000
N_PSILOQA = 20_000
N_VITAMINC = 24_000
SEED = 0

BARS = {"gold": (0.7095, 0.76), "ragtruth_en": (0.7039, 0.75), "ragtruth_nonen": (0.6095, 0.66)}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")

# spans convention per pair: None -> no token supervision (mask all);
# [] -> supervised, fully grounded (all claim tokens 0); [(s,e),...] -> in-span 1.


def private_train():
    """Soft teacher labels, TRAIN traces only - same seed-0 trace split as our_gold."""
    df = pl.read_parquet(PAIRS)
    g = pl.read_parquet(GOLD).with_row_index("owner")
    df = df.join(g.select(["owner", "trace_id"]), on="owner", how="left")
    traces = np.array(sorted(set(df["trace_id"].to_list())))
    rng = np.random.default_rng(SEED)
    rng.shuffle(traces)
    n_test, n_val = int(len(traces) * 0.25), int(len(traces) * 0.15)
    held = set(traces[: n_test + n_val].tolist())
    tr = df.filter(~pl.col("trace_id").is_in(list(held)))
    if len(tr) > N_PRIVATE:
        tr = tr.sample(N_PRIVATE, seed=SEED)
    y = tr["rerank"].to_numpy().astype("float32")
    return tr["claim"].to_list(), tr["chunk"].to_list(), y, [None] * len(y)


def public_train():
    """The H84 mix with per-pair span supervision where it exists."""
    claims, chunks, ys, spans = [], [], [], []

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
    df = df.sample(min(N_PUBLIC_EN, len(df)), seed=SEED)
    for out_txt, ctx, lab, raw in zip(
        df["output"].to_list(),
        df["context"].to_list(),
        df["label"].to_list(),
        df["hallucination_labels"].to_list(),
        strict=True,
    ):
        sp = []
        if raw:
            try:
                sp = [(int(d["start"]), int(d["end"])) for d in json.loads(raw)]
            except (ValueError, KeyError, TypeError):
                sp = []
        claims.append(out_txt)
        chunks.append(ctx[: M59.CFG.chunk_max_chars])
        ys.append(lab)
        spans.append(sp)

    zt = zipfile.ZipFile(DATA / "dataset-ragtruth-translated.zip")
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        nm = next(
            x for x in zt.namelist() if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet")
        )
        d = pl.read_parquet(io.BytesIO(zt.read(nm)))
        d = d.with_columns((pl.col("labels").list.len() == 0).cast(pl.Float32).alias("label"))
        d = d.filter(pl.col("prompt").str.len_chars() > 50)
        d = d.sample(min(N_PUBLIC_PER_LANG, len(d)), seed=SEED)
        for ans, prm, lab, ls in zip(
            d["answer"].to_list(),
            d["prompt"].to_list(),
            d["label"].to_list(),
            d["labels"].to_list(),
            strict=True,
        ):
            claims.append(ans)
            chunks.append(prm[: M59.CFG.chunk_max_chars])
            ys.append(lab)
            spans.append([(int(s["start"]), int(s["end"])) for s in (ls or [])])

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
        d = d.sample(min(N_HALUEVAL, len(d)), seed=SEED)
        for ev, pos, neg in zip(
            d[ev_col].to_list(), d[pos_col].to_list(), d[neg_col].to_list(), strict=True
        ):
            ev = ev[: M59.CFG.chunk_max_chars]
            claims += [pos, neg]
            chunks += [ev, ev]
            ys += [1.0, 0.0]
            spans += [[], [(0, len(neg))]]  # whole-answer span on the hallucinated twin

    zp = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    dp = pl.read_parquet(
        io.BytesIO(zp.read(next(x for x in zp.namelist() if x.endswith("__train.parquet"))))
    )
    dp = dp.filter(
        (pl.col("wiki_passage").str.len_chars() > 50) & (pl.col("llm_answer").str.len_chars() > 10)
    ).with_columns((pl.col("labels").list.len() == 0).cast(pl.Float32).alias("label"))
    dp = dp.sample(min(N_PSILOQA, len(dp)), seed=SEED)
    for ans, psg, lab, ls in zip(
        dp["llm_answer"].to_list(),
        dp["wiki_passage"].to_list(),
        dp["label"].to_list(),
        dp["labels"].to_list(),
        strict=True,
    ):
        claims.append(ans)
        chunks.append(psg[: M59.CFG.chunk_max_chars])
        ys.append(lab)
        spans.append([(int(p[0]), int(p[1])) for p in (ls or [])])

    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    dv = pl.read_parquet(
        io.BytesIO(zv.read(next(x for x in zv.namelist() if x.endswith("__train.parquet"))))
    )
    dv = dv.with_columns(
        (pl.col("label").cast(pl.Utf8).str.to_uppercase() == "SUPPORTS")
        .cast(pl.Float32)
        .alias("y")
    )
    dv = dv.sample(min(N_VITAMINC, len(dv)), seed=SEED)
    claims += dv["claim"].to_list()
    chunks += [c[: M59.CFG.chunk_max_chars] for c in dv["evidence"].to_list()]
    ys += dv["y"].to_list()
    spans += [None] * len(dv)  # atomic claims: wrong claim, not hallucinated span

    return claims, chunks, np.array(ys, dtype="float32"), spans


class TwoHeadSet(Dataset):
    def __init__(self, claims, chunks, y, spans, tok):
        self.c, self.k, self.y, self.s, self.tok = claims, chunks, y, spans, tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i], self.s[i]

    def collate(self, b):
        c, k, y, sp = zip(*b, strict=True)
        enc = self.tok(
            list(c),
            list(k),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        tok_labels = torch.full(offsets.shape[:2], -100, dtype=torch.long)
        for i, spans_i in enumerate(sp):
            if spans_i is None:
                continue  # no token supervision for this pair
            seq_ids = enc.sequence_ids(i)
            for j, sid in enumerate(seq_ids):
                if sid != 0:
                    continue  # only the claim (sequence A) is tagged
                a, bnd = offsets[i, j].tolist()
                if a == bnd:
                    continue
                inside = any(not (bnd <= s or a >= e) for s, e in spans_i)
                tok_labels[i, j] = 1 if inside else 0
        return enc, torch.tensor(y), tok_labels


class TwoHeadStudent(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.trunk = base
        d = base.config.hidden_size
        self.score_head = nn.Linear(d, 1)
        self.token_head = nn.Linear(d, 2)

    def forward(self, enc):
        h = self.trunk(**enc).last_hidden_state
        return self.score_head(h[:, 0]).squeeze(-1), self.token_head(h)


@torch.inference_mode()
def score_student(model, tok, claims, chunk_lists):
    """Fused per-pair prob = (p_score + p_token)/2, then max-over-chunks.

    Also returns the per-claim per-head scores so orthogonality inside the
    trunk is measurable.
    """
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k)
            owner.append(i)
    p_sc = np.zeros(len(flat_c), dtype=np.float32)
    p_tk = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), 64):
        enc = tok(
            flat_c[i : i + 64],
            flat_k[i : i + 64],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        seq0 = torch.tensor(
            [
                [enc.sequence_ids(r)[j] == 0 for j in range(enc["input_ids"].shape[1])]
                for r in range(enc["input_ids"].shape[0])
            ]
        )
        enc = {k: v.cuda() for k, v in enc.items()}
        s_logit, t_logit = model(enc)
        p_sc[i : i + 64] = torch.sigmoid(s_logit.float()).cpu().numpy()
        halluc = torch.softmax(t_logit.float(), dim=-1)[:, :, 1].cpu()
        halluc[~seq0] = 0.0  # only claim tokens vote
        p_tk[i : i + 64] = (1.0 - halluc.max(dim=1).values).numpy()
    owner = np.array(owner)
    fused = (p_sc + p_tk) / 2.0
    agg = lambda v: np.array([v[owner == i].max() for i in range(len(claims))])
    return agg(fused), agg(p_sc), agg(p_tk)


def evaluate(model, tok):
    """Three-corpus gate; fused is the headline, per-head recorded alongside."""
    from scipy.stats import spearmanr

    res = {}
    sub = _mod("sub", "R8_score_substrate.py")

    claims, chunk_lists, y, _, _ = sub.our_gold()
    f, s, t = score_student(model, tok, claims, chunk_lists)
    auc, f1, _ = M59.auc_and_f1(y, f)
    res["gold"] = {
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "n": len(y),
        "auc_score_head": round(M59.auc_and_f1(y, s)[0], 4),
        "auc_token_head": round(M59.auc_and_f1(y, t)[0], 4),
        "head_spearman": round(float(spearmanr(s, t).statistic), 4),
    }

    cl, ctx, y = M60.load_english()
    chunks = [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
    f, s, t = score_student(model, tok, cl, chunks)
    auc, f1, _ = M59.auc_and_f1(y, f)
    res["ragtruth_en"] = {
        "auc": round(auc, 4),
        "f1": round(f1, 4),
        "n": len(y),
        "auc_score_head": round(M59.auc_and_f1(y, s)[0], 4),
        "auc_token_head": round(M59.auc_and_f1(y, t)[0], 4),
        "head_spearman": round(float(spearmanr(s, t).statistic), 4),
    }

    per_lang = {}
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        cl, ctx, y = M60.load_translated(lg)
        f, _, _ = score_student(
            model, tok, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
        )
        auc, f1, _ = M59.auc_and_f1(y, f)
        per_lang[lg] = round(auc, 4)
    res["ragtruth_nonen"] = {
        "auc": round(float(np.mean(list(per_lang.values()))), 4),
        "per_lang": per_lang,
    }
    return res


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    pc, pk, py, psp = private_train()
    uc, uk, uy, usp = public_train()
    claims, chunks = pc + uc, pk + uk
    y = np.concatenate([py, uy])
    spans = psp + usp
    n_tok = sum(1 for s in spans if s is not None)
    n_spanned = sum(1 for s in spans if s)
    print(
        f"train: {len(y)} pairs, mean target {y.mean():.3f}; token supervision on "
        f"{n_tok} pairs ({n_spanned} carry at least one span)\n",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(STUDENT)
    base = AutoModel.from_pretrained(STUDENT).cuda()
    model = TwoHeadStudent(base).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {STUDENT} + two heads  {n_par:.1f}M params  (ceiling 400M)\n", flush=True)

    ds = TwoHeadSet(claims, chunks, y, spans, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=ds.collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=len(dl), pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    score_lossf = nn.BCEWithLogitsLoss()
    token_lossf = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    t0 = time.time()
    for step, (enc, yy, tl) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, tl = yy.cuda(), tl.cuda()
        s_logit, t_logit = model(enc)
        s_loss = score_lossf(s_logit, yy)
        if (tl != -100).any():
            t_loss = token_lossf(t_logit.reshape(-1, 2), tl.reshape(-1))
        else:
            t_loss = torch.zeros((), device=s_logit.device)
        loss = s_loss + W_TOK * t_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()
        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            print(
                f"  step {step}/{len(dl)} score {s_loss.item():.4f} "
                f"token {t_loss.item():.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trunk": base.state_dict(),
            "score_head": model.score_head.state_dict(),
            "token_head": model.token_head.state_dict(),
        },
        CKPT_DIR / "twohead.pt",
    )
    tok.save_pretrained(CKPT_DIR)
    base.save_pretrained(CKPT_DIR / "trunk")
    print(f"\ncheckpoint saved -> {CKPT_DIR}\n", flush=True)

    model.eval()
    res = evaluate(model, tok)
    res["params_M"] = round(n_par, 1)
    res["w_tok"] = W_TOK

    print("=" * 96)
    print("R8-H73 RESULT - two-head trunk (fused), one sub-400M model")
    print("=" * 96)
    print(f"{'corpus':18s} {'fused':>9} {'bar':>9} {'delta':>9} {'decisive':>10}")
    won = 0
    for key, (bar, decisive) in BARS.items():
        a = res[key]["auc"]
        mark = "DECISIVE" if a >= decisive else ("beat" if a > bar else "LOSE")
        won += a > bar
        print(f"{key:18s} {a:>9.4f} {bar:>9.4f} {a - bar:>+9.4f} {decisive:>10.4f}  {mark}")
    for key in ("gold", "ragtruth_en"):
        r = res[key]
        print(
            f"  {key}: score-head {r['auc_score_head']} token-head {r['auc_token_head']} "
            f"head-spearman {r['head_spearman']}"
        )
    print(f"\n  corpora beaten: {won}/3   params {n_par:.1f}M / 400M")
    print("  blind arena: score separately via R8-H77 --model (twohead-aware)")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
