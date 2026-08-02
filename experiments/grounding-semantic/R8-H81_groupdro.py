"""R8-H81 - GroupDRO: minimise the WORST-domain loss, not the average.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 8).
Runs FIRST of the adversarial arm - no discriminator, no lambda, and it targets
the measured failure most directly.

R8-H77 measured a mean that hides a collapse: 0.5956 against the incumbent's
0.6461, with `finqa` below chance at 0.3974. That is a literal description of a
loss being AVERAGED over domains, so a domain the model handles badly is drowned
out by nine it handles well. GroupDRO (Sagawa et al. 2020) makes a different
assumption than the adversarial framing - the problem is not that features
ENCODE domain (that is R8-H79's bet) but that the objective flattens over them.
It optimises the worst group directly via exponentiated-gradient group weights
q_g, needs no gradient reversal, no discriminator and no lambda schedule, and
cannot collapse the representation.

Lever - the loss aggregation only. The recipe, the data mix and the data itself
are byte-identical to R8-H84 (the best blind-arena mix that leaves RAGBench
untouched), so any delta is attributable to the worst-group weighting and
nothing else. Groups are the corpus of origin: private / ragtruth_en /
ragtruth_<lang> / halueval / psiloqa / vitaminc.

The one implementation constraint GroupDRO imposes: group weights are updated
from MEAN PER-GROUP LOSS, so a batch must contain every group or the unseen
groups are starved of signal. A shuffled 100k-row loader cannot guarantee that.
This run therefore buckets by group and draws a balanced stratified batch each
step - every group present at equal count - which is the standard remedy and is
declared here rather than discovered afterwards. Steps are set so the sampler
sees ~1 epoch of the data.

Re-registered bar (the arena bar tracks the best blind mean achieved without
touching it): blind-arena mean >= 0.6450 with worst-subset AUC >= 0.55 and our
gold holding >= 0.84. The blind arena is scored separately through the R8-H77
`--model` gate, exactly as H83 and H84 were.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R8-H81_groupdro.py
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
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent.parent / "data" / "external" / "datasets"
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
CKPT_DIR = HERE.parent.parent / "models" / "R8-H81-mmbert-groupdro"
OUT = HERE / "R8-H81_result.json"

STUDENT = "jhu-clsp/mmBERT-base"  # 307M, under the 400M ceiling, same size as the incumbent
MAX_LEN = 512
BATCH = 48  # >= 4x the 11 groups so the stratified sampler yields full batches
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
N_PRIVATE = 40_000
N_PUBLIC_EN = 15_000
N_PUBLIC_PER_LANG = 4_000
N_HALUEVAL = 6_000  # per config; each row yields TWO claims, one of each class
N_PSILOQA = 20_000
N_VITAMINC = 24_000  # sampled; keeps one stratified epoch to a tractable step count
SEED = 0

# GroupDRO: exponentiated-gradient group weights. ETA_Q is the step size for the
# q_g update; the recipe's only new hyperparameter, and deliberately not swept
# in this run (one clean read against the bar before any tuning).
ETA_Q = 0.01
GROUP_WEIGHT_DECAY = 0.01  # strong regularisation, which GroupDRO requires

BARS = {"gold": (0.7095, 0.76), "ragtruth_en": (0.7039, 0.75), "ragtruth_nonen": (0.6095, 0.66)}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")


def private_train():
    """Soft teacher labels, TRAIN traces only - the held-out 159 never appear.

    The split is the SAME seed-0 trace shuffle the substrate's our_gold uses,
    holding out the first 40% of traces (a superset of the substrate's 25% test)
    so the gold gate stays disjoint. Do not migrate to R8_splits.frames here
    without migrating the substrate too - two drifting split definitions caused
    the round-7 leak.
    """
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
    claims = tr["claim"].to_list()
    chunks = tr["chunk"].to_list()
    y = tr["rerank"].to_numpy().astype("float32")
    return claims, chunks, y, ["private"] * len(y)


def public_train():
    """Hard human labels from RAGTruth, HaluEval, PsiloQA, VitaminC - the H84 mix.

    RAGBench is deliberately EXCLUDED - it is the blind arena, and training on
    it is the recorded R8-H78 error. Returns a parallel domain tag per pair.
    """
    claims, chunks, ys, tags = [], [], [], []

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
    tags += ["ragtruth_en"] * len(df)

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
        tags += [f"ragtruth_{lg}"] * len(d)

    # HaluEval - matched positive and negative from the SAME evidence.
    zh = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    n_hal = 0
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
            tags += ["halueval", "halueval"]
            n_hal += 2

    # PsiloQA - 14 languages.
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
    tags += ["psiloqa"] * len(dp)

    # VitaminC - near-miss negatives: a single factual edit flips the verdict.
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
    dv = dv.sample(min(N_VITAMINC, len(dv)), seed=SEED)
    claims += dv[cl_col].to_list()
    chunks += [c[: M59.CFG.chunk_max_chars] for c in dv[ev_col].to_list()]
    ys.append(dv["y"].to_numpy())
    tags += ["vitaminc"] * len(dv)

    return claims, chunks, np.concatenate(ys).astype("float32"), tags


class GroupSet(Dataset):
    """Pairs carrying a group index; collate returns (enc, y, group)."""

    def __init__(self, claims, chunks, y, groups, tok):
        self.c, self.k, self.y, self.g, self.tok = claims, chunks, y, groups, tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i], self.g[i]

    def collate(self, b):
        c, k, y, g = zip(*b, strict=True)
        enc = self.tok(
            list(c),
            list(k),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        return enc, torch.tensor(y), torch.tensor(g)


class StratifiedGroupSampler:
    """Balanced per-group batches so every group updates q_g every step.

    GroupDRO updates group weights from mean per-group loss; a shuffled loader
    over a 100k-row mix cannot guarantee every group appears in every 32-row
    batch, so the absent groups' q_g would decay toward uniform and the
    worst-group signal would be lost. This sampler draws BATCH/n_groups rows
    from every group each step.
    """

    def __init__(self, groups, batch, n_groups, seed):
        self.by_group = [np.where(groups == g)[0] for g in range(n_groups)]
        self.per = batch // n_groups
        self.batch = self.per * n_groups
        self.n_groups = n_groups
        self.rng = np.random.default_rng(seed)

    def batches(self, n_steps):
        for _ in range(n_steps):
            idx = np.concatenate(
                [self.rng.choice(b, size=self.per, replace=True) for b in self.by_group]
            )
            self.rng.shuffle(idx)
            yield idx


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
    pc, pk, py, ptags = private_train()
    uc, uk, uy, utags = public_train()
    claims, chunks = pc + uc, pk + uk
    y = np.concatenate([py, uy])
    tag_names = sorted(set(ptags + utags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in ptags + utags])
    n_groups = len(tag_names)
    print(
        f"train: {len(y)} pairs across {n_groups} groups, mean target {y.mean():.3f}",
        flush=True,
    )
    counts = {t: int((groups == i).sum()) for t, i in tag_to_idx.items()}
    print(f"  group sizes: {counts}\n", flush=True)

    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=1, ignore_mismatched_sizes=True
    ).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {STUDENT}  {n_par:.1f}M params  (ceiling 400M)\n", flush=True)

    ds = GroupSet(claims, chunks, y, groups, tok)
    sampler = StratifiedGroupSampler(groups, BATCH, n_groups, SEED)
    # One stratified pass over the data.
    n_steps = len(y) // sampler.batch
    collate = ds.collate

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=GROUP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    # Exponentiated-gradient group weights, initialised uniform.
    q = torch.full((n_groups,), 1.0 / n_groups).cuda()

    model.train()
    t0 = time.time()
    gen = sampler.batches(n_steps)
    for step in range(n_steps):
        idx = next(gen)
        batch = [ds[int(i)] for i in idx]
        enc, yy, gg = collate(batch)
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg = yy.cuda(), gg.cuda()
        per_loss = lossf(model(**enc).logits.squeeze(-1), yy)

        # GroupDRO: mean loss per group, then the q_g-weighted worst-group loss.
        group_loss = torch.zeros(n_groups).cuda()
        for g in range(n_groups):
            m = gg == g
            if m.any():
                group_loss[g] = per_loss[m].mean()
        loss = (q.detach() * group_loss).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()

        # Exponentiated-gradient update of the group weights.
        q = q * torch.exp(ETA_Q * group_loss.detach())
        q = q / q.sum()

        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            worst = group_loss.max().item()
            qd = {tag_names[g]: round(q[g].item(), 3) for g in range(n_groups)}
            print(
                f"  step {step}/{n_steps} worst-group loss {worst:.4f} "
                f"({time.time() - t0:.0f}s)  q={qd}",
                flush=True,
            )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    tok.save_pretrained(CKPT_DIR)
    print(f"\ncheckpoint saved -> {CKPT_DIR}\n", flush=True)

    model.eval()
    res = evaluate(model, tok)
    res["params_M"] = round(n_par, 1)
    res["final_group_weights"] = {tag_names[g]: round(q[g].item(), 4) for g in range(n_groups)}

    print("=" * 96)
    print("R8-H81 RESULT - GroupDRO worst-domain, one sub-400M model")
    print("=" * 96)
    print(f"{'corpus':18s} {'ours':>9} {'lettuce':>9} {'delta':>9} {'decisive bar':>13} {'':>6}")
    won = 0
    for key, (bar, decisive) in BARS.items():
        a = res[key]["auc"]
        mark = "DECISIVE" if a >= decisive else ("beat" if a > bar else "LOSE")
        won += a > bar
        print(f"{key:18s} {a:>9.4f} {bar:>9.4f} {a - bar:>+9.4f} {decisive:>13.4f}  {mark}")
    print(f"\n  corpora beaten: {won}/3   params {n_par:.1f}M / 400M")
    print(f"  final group weights: {res['final_group_weights']}")
    print("  blind arena: score separately via R8-H77 --model")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
