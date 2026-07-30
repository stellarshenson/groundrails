"""R8-H84 - incarnation 4: near-miss negatives from VitaminC.

R8-H83 refuted the diversity hypothesis: 44k rows of HaluEval and PsiloQA moved
the blind arena +0.0205 to 0.6161, still 0.0300 below the incumbent, while 30k
rows of the actual test domain had moved it +0.1085. Generic diversity buys
about a fifth of what in-domain data buys, from more rows.

But the per-subset breakdown says the failure is not really about domain. We win
every BALANCED subset - expertqa (base 0.468) +0.0709, techqa (0.564) +0.0275,
covidqa (0.841) +0.0171 - and lose every extreme-base-rate one, where a handful
of negatives decide the AUC. That is a NEGATIVE-CLASS problem.

Every corpus in the mix so far carries topically DISTANT negatives at base rates
0.65-0.94, so a model can score well by learning the prior rather than the
boundary. VitaminC is the only corpus in the survey built the other way: real
Wikipedia revisions where a single factual edit flips the verdict, so a negative
differs from its positive by one number, entity or qualifier and nothing else.
370,653 train rows, SUPPORTS 185,714 / REFUTES 131,958 / NOT ENOUGH INFO 52,981 -
close to balanced, which nothing else here is.

Three-way labels collapse to binary: SUPPORTS -> grounded, REFUTES and NEI -> not.

RAGBench remains excluded, so the arena stays a genuine blind test.

Original header follows.

R8-H83 - incarnation 3: domain diversity WITHOUT touching the blind test.

R8-H78 proved the mechanism - adding RAGBench train took the blind arena from
0.5956 to 0.7041 and `finqa` from 0.3974 (below chance) to 0.7433. But it bought
that by training on the test set's own train split, which makes RAGBench no
longer blind for us while the incumbent still has not seen it. That comparison
is worthless as evidence and was recorded as such.

This incarnation reproduces the DIVERSITY gain from sources that leave RAGBench
untouched, so the arena stays a genuine blind test:

  - HaluEval (MIT, ~24.5k) - `knowledge` plus BOTH a `right_answer` and a
    `hallucinated_answer` from identical evidence, so it contributes perfectly
    balanced pairs rather than the 0.65 base rate everything else carries
  - PsiloQA (CC-BY-4.0, 63.8k, 14 languages) - `wiki_passage` and `llm_answer`
    with character spans; en fi hi ca sv zh cs fa es it eu ar fr de, which is
    six more languages than RAGTruth's translations reach

Deliberately EXCLUDED:
  - RAGBench - it is the blind arena, and training on it is the error this
    incarnation exists to undo
  - LettuceDetect-prose - it is the incumbent's own training set AND is derived
    from PsiloQA (its rows carry `dataset: psiloqa`), so including it would both
    double-count the upstream and make our corpus a literal superset of theirs.
    PsiloQA is the shared upstream and is fair to both

Original header follows.

R8-H62 - multi-corpus distillation: one sub-400M model for all three corpora.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).

The win condition is a single model under 400M parameters strictly above
`lettucedect-v2-mmbert-base` on all three corpora at once:

    private gold      beat 0.7095, decisive >= 0.76
    RAGTruth EN       beat 0.7039, decisive >= 0.75
    RAGTruth non-EN   beat 0.6095, decisive >= 0.66

Round 7 already produced half of that by accident: a mmBERT-base student
distilled from our cascade reads 0.8479 on gold against their 0.7095, at the
same 307M parameter count. What it never measured is that student on the other
two corpora - and R7-H50 deleted every checkpoint after scoring it, so the
weights are gone. This run fixes both: it trains, SAVES, and evaluates on all
three.

Two supervision signals, deliberately mixed:

  - OUR corpus contributes SOFT labels - the reranker's calibrated per-pair
    score. Soft targets carry the teacher's uncertainty and are what let a
    student generalise beyond hard labels
  - the PUBLIC corpora contribute HARD labels - RAGTruth's human span
    annotations, collapsed to response-level grounded/not

Both point the same way (1 = grounded), so they share one head and one loss. The
mixing ratio is the lever: too much public data and the +0.152 domain advantage
erodes, too little and the other two objectives do not move. This run fixes it
at roughly 1:1 and R8-H62b sweeps it if the first attempt lands close.

Training uses ONLY train splits - RAGTruth test and our held-out traces are
never seen, and the evaluation reuses the exact harness from R7-H59 / R7-H60 so
the numbers are directly comparable to the bars above.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R8-H62_multicorpus_distill.py
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
CKPT_DIR = HERE.parent.parent / "models" / "R8-H84-mmbert-vitaminc"
OUT = HERE / "R8-H84_result.json"

STUDENT = "jhu-clsp/mmBERT-base"  # 307M, under the 400M ceiling, same size as the incumbent
MAX_LEN = 512
BATCH = 32
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
N_PRIVATE = 40_000
N_PUBLIC_EN = 15_000
N_PUBLIC_PER_LANG = 4_000
N_HALUEVAL = 6_000  # per config; each row yields TWO claims, one of each class
N_PSILOQA = 20_000  # sampled across all 14 languages
N_VITAMINC = 40_000  # near-miss negatives - the only near-balanced source in the mix
SEED = 0

BARS = {"gold": (0.7095, 0.76), "ragtruth_en": (0.7039, 0.75), "ragtruth_nonen": (0.6095, 0.66)}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")


def private_train():
    """Soft teacher labels, TRAIN traces only - the held-out 159 never appear."""
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
    return tr["claim"].to_list(), tr["chunk"].to_list(), tr["rerank"].to_numpy().astype("float32")


def public_train():
    """Hard human labels from RAGTruth TRAIN splits, English plus 7 translations."""
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
    df = df.sample(min(N_PUBLIC_EN, len(df)), seed=SEED)
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
        d = d.sample(min(N_PUBLIC_PER_LANG, len(d)), seed=SEED)
        claims += d["answer"].to_list()
        chunks += [c[: M59.CFG.chunk_max_chars] for c in d["prompt"].to_list()]
        ys.append(d["label"].to_numpy())

    # HaluEval - the only corpus here that supplies a matched positive and
    # negative from the SAME evidence, so it teaches the boundary rather than
    # the prior. Both configs carry a knowledge/document field; dialogue and
    # general do not and are skipped.
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
            print(f"    SKIP halueval/{cfg}: columns are {d.columns}", flush=True)
            continue
        d = d.sample(min(N_HALUEVAL, len(d)), seed=SEED)
        for ev, pos, neg in zip(
            d[ev_col].to_list(), d[pos_col].to_list(), d[neg_col].to_list(), strict=True
        ):
            ev = ev[: M59.CFG.chunk_max_chars]
            claims += [pos, neg]
            chunks += [ev, ev]
            ys.append(np.array([1.0, 0.0], dtype="float32"))

    # PsiloQA - 14 languages, six more than RAGTruth reaches. An empty `labels`
    # list means no hallucinated span was annotated, i.e. grounded.
    zp = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    dp = pl.read_parquet(
        io.BytesIO(zp.read(next(x for x in zp.namelist() if x.endswith("__train.parquet"))))
    )
    dp = dp.filter(
        (pl.col("wiki_passage").str.len_chars() > 50) & (pl.col("llm_answer").str.len_chars() > 10)
    ).with_columns((pl.col("labels").list.len() == 0).cast(pl.Float32).alias("label"))
    dp = dp.sample(min(N_PSILOQA, len(dp)), seed=SEED)
    claims += dp["llm_answer"].to_list()
    chunks += [c[: M59.CFG.chunk_max_chars] for c in dp["wiki_passage"].to_list()]
    ys.append(dp["label"].to_numpy())

    # VitaminC - the only NEAR-MISS negatives available. Contrastive Wikipedia
    # revisions where one factual edit flips the verdict, so the negative is
    # minimally different from its positive rather than topically distant. This
    # targets the R8-H83 diagnosis directly: we lose exactly where the negative
    # class is thin and the boundary, not the prior, decides the AUC.
    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    dv = pl.read_parquet(
        io.BytesIO(zv.read(next(x for x in zv.namelist() if x.endswith("__train.parquet"))))
    )
    dv = dv.filter(
        (pl.col("evidence").str.len_chars() > 20) & (pl.col("claim").str.len_chars() > 10)
    ).with_columns((pl.col("label") == "SUPPORTS").cast(pl.Float32).alias("bin"))
    dv = dv.sample(min(N_VITAMINC, len(dv)), seed=SEED)
    claims += dv["claim"].to_list()
    chunks += [c[: M59.CFG.chunk_max_chars] for c in dv["evidence"].to_list()]
    ys.append(dv["bin"].to_numpy())

    return claims, chunks, np.concatenate(ys).astype("float32")


class PairSet(Dataset):
    def __init__(self, claims, chunks, y, tok):
        self.c, self.k, self.y, self.tok = claims, chunks, y, tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i]

    def collate(self, b):
        c, k, y = zip(*b, strict=True)
        enc = self.tok(
            list(c),
            list(k),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        return enc, torch.tensor(y)


@torch.inference_mode()
def score_student(model, tok, claims, chunk_lists):
    """Max-over-chunks, exactly as the cascade serves."""
    flat_c, flat_k, owner = [], [], []
    for i, (c, ks) in enumerate(zip(claims, chunk_lists, strict=True)):
        for k in ks:
            flat_c.append(c)
            flat_k.append(k)
            owner.append(i)
    out = np.zeros(len(flat_c), dtype=np.float32)
    for i in range(0, len(flat_c), 64):
        enc = tok(
            flat_c[i : i + 64],
            flat_k[i : i + 64],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(model.device)
        out[i : i + 64] = torch.sigmoid(model(**enc).logits.float().squeeze(-1)).cpu().numpy()
    owner = np.array(owner)
    return np.array([out[owner == i].max() for i in range(len(claims))])


def evaluate(model, tok):
    """All three corpora, same harness and metric as the recorded bars."""
    res = {}

    claims, chunk_lists, y, _, _ = _mod("sub", "R8_score_substrate.py").our_gold()
    s = score_student(model, tok, claims, chunk_lists)
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["gold"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    cl, ctx, y = M60.load_english()
    s = score_student(model, tok, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx])
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["ragtruth_en"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    per_lang = {}
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        cl, ctx, y = M60.load_translated(lg)
        s = score_student(model, tok, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx])
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
    print(
        f"train: {len(pc)} private (soft teacher labels) + {len(uc)} public (hard human "
        f"labels) = {len(y)} pairs, mean target {y.mean():.3f}\n",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=1, ignore_mismatched_sizes=True
    ).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {STUDENT}  {n_par:.1f}M params  (ceiling 400M)\n", flush=True)

    ds = PairSet(claims, chunks, y, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=ds.collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=len(dl), pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    lossf = torch.nn.BCEWithLogitsLoss()

    model.train()
    t0 = time.time()
    for step, (enc, yy) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        loss = lossf(model(**enc).logits.squeeze(-1), yy.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()
        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            print(
                f"  step {step}/{len(dl)} loss {loss.item():.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tok.save_pretrained(CKPT_DIR)
    print(
        f"\ncheckpoint saved -> {CKPT_DIR}  (R7-H50 deleted its students; this one persists)\n",
        flush=True,
    )

    model.eval()
    res = evaluate(model, tok)
    res["params_M"] = round(n_par, 1)

    print("=" * 96)
    print("R8-H62 RESULT - one sub-400M model, all three corpora")
    print("=" * 96)
    print(f"{'corpus':18s} {'ours':>9} {'lettuce':>9} {'delta':>9} {'decisive bar':>13} {'':>6}")
    won = 0
    for key, (bar, decisive) in BARS.items():
        a = res[key]["auc"]
        mark = "DECISIVE" if a >= decisive else ("beat" if a > bar else "LOSE")
        won += a > bar
        print(f"{key:18s} {a:>9.4f} {bar:>9.4f} {a - bar:>+9.4f} {decisive:>13.4f}  {mark}")
    print(f"\n  corpora beaten: {won}/3   params {n_par:.1f}M / 400M")
    print(f"  per-language: {res['ragtruth_nonen']['per_lang']}")
    if won == 3:
        print("  -> WIN CONDITION MET, one model above the incumbent on every corpus")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
