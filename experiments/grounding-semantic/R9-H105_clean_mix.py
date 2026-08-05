"""R9-H105 - clean-mix recipe baseline: the H90 recipe with the private pairs removed.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 9).

The author reset the training protocol on 2026-08-03: the private gold dataset
becomes test-only. `private_train()` (76,865 teacher pairs from the TRAIN split
of the 639 gold traces, ~10% of the H90 mix) leaves the mix entirely; training
is `public_train()` only - RAGTruth EN + 7 translations, HaluEval, PsiloQA,
VitaminC, TabFact - ~686k pairs, 12 domain groups (chance 1/12 ~ 0.083).

Everything else is the H90 recipe unchanged: mmBERT-base, BCE + DANN lambda
0.02 (Ganin ramp), MAX_LEN 512, BATCH 48, LR 1e-5, SEED 0. RAGBench remains
excluded - the arena stays blind.

Evaluation: the three lineage gates (our_gold split, ragtruth_en,
ragtruth_nonen) are kept with their recorded bars for comparability, and a new
`gold_full` read over ALL 2,752 gold claims is added - legal as a pure test set
under the clean-mix protocol, reported with NO bar (first clean measurement).

Bar: blind-arena mean >= 0.6461 (beat the incumbent); this draw re-establishes
the recipe baseline - its numbers become the reference for every future lever.
Kill: training instability only.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R9-H105_clean_mix.py
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
# Eval-only under the clean-mix protocol: PAIRS supplies each gold claim's
# chunk lists for the gold reads; no row of it enters training.
PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
CKPT_DIR = HERE.parent.parent / "models" / "R9-H105-mmbert-dann-clean"
OUT = HERE / "R9-H105_result.json"

STUDENT = "jhu-clsp/mmBERT-base"  # 307M, under the 400M ceiling, same size as the incumbent
MAX_LEN = 512
BATCH = 48
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
SEED = 0

# DANN: the adversarial weight, unchanged from the H90 recipe.
LAMBDA_MAX = 0.02
DANN_HIDDEN = 256

# Lineage bars (split-gate era, models then trained in-domain) - kept for
# comparability only; the clean model is expected to sit below the old gold
# decisive line since gold is now out-of-domain.
BARS = {"gold": (0.7095, 0.76), "ragtruth_en": (0.7039, 0.75), "ragtruth_nonen": (0.6095, 0.66)}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")


def public_train():
    """The H84 mix (RAGTruth, HaluEval, PsiloQA, VitaminC), RAGBench excluded, with
    a domain tag per pair for the discriminator."""
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
        claims += d["answer"].to_list()
        chunks += [c[: M59.CFG.chunk_max_chars] for c in d["prompt"].to_list()]
        ys.append(d["label"].to_numpy())
        tags += [f"ragtruth_{lg}"] * len(d)

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
            tags += ["halueval", "halueval"]

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
    tags += ["psiloqa"] * len(dp)

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
    tags += ["vitaminc"] * len(dv)

    # TabFact - the R8-H87 tabular-register lever, folded into the full mix.
    # Counterfactual statements against the SAME Wikipedia table (near-miss by
    # construction, human ENTAILED/REFUTED labels, ~55/45 balance). Evidence is
    # the caption plus the linearized table, `#` cells -> ` | ` for readability.
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
    tags += ["tabfact"] * len(dt)

    return claims, chunks, np.concatenate(ys).astype("float32"), tags


def gold_full():
    """ALL 2,752 gold claims - a pure test set under the clean-mix protocol (no
    trace enters training). Same per-claim construction as the substrate's
    our_gold - one row per claim carrying ALL of its chunks - minus the
    trace-split filter."""
    df = pl.read_parquet(PAIRS)
    claims, chunk_lists, labels = [], [], []
    for owner, grp in df.group_by("owner"):
        claims.append(grp["claim"][0])
        chunk_lists.append(grp["chunk"].to_list())
        labels.append(int(grp["label"][0]))
    return claims, chunk_lists, np.array(labels)


class GroupSet(Dataset):
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


class GradReverse(torch.autograd.Function):
    """Identity forward, negated-scaled backward - the gradient reversal layer."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class DANNStudent(nn.Module):
    """mmBERT trunk, a grounding head, and a domain discriminator off the GRL.

    The grounding head is a linear layer to one logit (the task); the domain
    discriminator is an MLP over the GRL'd [CLS] to n_groups logits. Both read
    the same pooled feature, which is what makes the adversarial game well-posed.
    """

    def __init__(self, base, n_groups, hidden=DANN_HIDDEN):
        super().__init__()
        self.trunk = base
        d = base.config.hidden_size
        self.pool = nn.Identity()
        self.task_head = nn.Linear(d, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_groups)
        )

    def forward(self, enc, lam):
        out = self.trunk(**enc).last_hidden_state[:, 0]  # [CLS]
        task_logit = self.task_head(out).squeeze(-1)
        domain_logit = self.domain_head(GradReverse.apply(out, lam))
        return task_logit, domain_logit


@torch.inference_mode()
def score_student(trunk, task_head, tok, claims, chunk_lists):
    """Max-over-chunks over the task head, exactly as the cascade serves."""
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
        ).to(trunk.device)
        cls = trunk(**enc).last_hidden_state[:, 0]
        out[i : i + 64] = torch.sigmoid(task_head(cls).float().squeeze(-1)).cpu().numpy()
    owner = np.array(owner)
    return np.array([out[owner == i].max() for i in range(len(claims))])


def evaluate(model, tok):
    """The three lineage gates plus the full-gold read (clean-mix protocol)."""
    trunk, task_head = model.trunk, model.task_head
    res = {}

    claims, chunk_lists, y, _, _ = _mod("sub", "R8_score_substrate.py").our_gold()
    s = score_student(trunk, task_head, tok, claims, chunk_lists)
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["gold"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    cl_f, ck_f, y_f = gold_full()
    s = score_student(trunk, task_head, tok, cl_f, ck_f)
    auc, f1, _ = M59.auc_and_f1(y_f, s)
    res["gold_full"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y_f)}

    cl, ctx, y = M60.load_english()
    s = score_student(
        trunk, task_head, tok, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
    )
    auc, f1, _ = M59.auc_and_f1(y, s)
    res["ragtruth_en"] = {"auc": round(auc, 4), "f1": round(f1, 4), "n": len(y)}

    per_lang = {}
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        cl, ctx, y = M60.load_translated(lg)
        s = score_student(
            trunk, task_head, tok, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
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
    claims, chunks, y, tags = public_train()
    tag_names = sorted(set(tags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(tag_names)
    chance = 1.0 / n_groups
    print(
        f"train: {len(y)} pairs across {n_groups} domains (chance {chance:.3f}), "
        f"mean target {y.mean():.3f}  [clean mix - no private pairs]\n",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(STUDENT)
    base = AutoModel.from_pretrained(STUDENT).cuda()
    model = DANNStudent(base, n_groups).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {STUDENT} + DANN heads  {n_par:.1f}M params  (ceiling 400M)\n", flush=True)

    ds = GroupSet(claims, chunks, y, groups, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=ds.collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    task_lossf = nn.BCEWithLogitsLoss()
    domain_lossf = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for step, (enc, yy, gg) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg = yy.cuda(), gg.cuda()
        # Ganin ramp: lambda grows from 0 to LAMBDA_MAX over training.
        p = step / max(n_steps - 1, 1)
        lam = LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        task_logit, domain_logit = model(enc, lam)
        t_loss = task_lossf(task_logit, yy)
        d_loss = domain_lossf(domain_logit, gg)
        loss = t_loss + d_loss  # d_loss is already scaled by lam inside the GRL

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad()

        dom_correct += (domain_logit.argmax(-1) == gg).sum().item()
        dom_total += len(gg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"diverged at step {step}")
        if step % 200 == 0:
            acc = dom_correct / max(dom_total, 1)
            dom_correct, dom_total = 0, 0
            print(
                f"  step {step}/{n_steps} task {t_loss.item():.4f} domain {d_loss.item():.4f} "
                f"lam {lam:.4f}  domain-acc {acc:.3f} (chance {chance:.3f})  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # Persist trunk + both heads; the task head is what the arena scores.
    torch.save(
        {
            "trunk": base.state_dict(),
            "task_head": model.task_head.state_dict(),
            "domain_head": model.domain_head.state_dict(),
            "config": base.config,
        },
        CKPT_DIR / "dann_student.pt",
    )
    tok.save_pretrained(CKPT_DIR)
    base.save_pretrained(CKPT_DIR / "trunk")
    print(f"\ncheckpoint saved -> {CKPT_DIR}\n", flush=True)

    model.eval()
    res = evaluate(model, tok)
    res["params_M"] = round(n_par, 1)
    res["lambda_max"] = LAMBDA_MAX
    res["mix"] = "clean (public only, no private pairs)"

    print("=" * 96)
    print("R9-H105 RESULT - clean-mix recipe baseline (private pairs removed)")
    print("=" * 96)
    print(f"{'corpus':18s} {'ours':>9} {'lineage bar':>11} {'delta':>9} {'decisive':>9} {'':>6}")
    won = 0
    for key, (bar, decisive) in BARS.items():
        a = res[key]["auc"]
        mark = "DECISIVE" if a >= decisive else ("beat" if a > bar else "LOSE")
        won += a > bar
        print(f"{key:18s} {a:>9.4f} {bar:>11.4f} {a - bar:>+9.4f} {decisive:>9.4f}  {mark}")
    gf = res["gold_full"]
    print(
        f"{'gold_full':18s} {gf['auc']:>9.4f} {'no bar':>11s} {'':>9} {'':>9}  "
        f"first clean measurement (n={gf['n']})"
    )
    print(f"\n  lineage corpora beaten: {won}/3   params {n_par:.1f}M / 400M")
    print("  blind arena: score separately via R8-H77 --model")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
