"""R9-H106 - clean-mix two-head with DANN, served with post-aggregation fusion.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 9).

Precursor P-A proved the mechanism deterministically on the frozen H102
checkpoint: fusing the two heads AFTER each one's windowed decomposed-min
(parameter-free logit-mean at the response level) read 0.7223 blind against
0.7172 (score) / 0.7051 (token). This draw retrains the same two-head recipe on
the CLEAN mix - the H102 trainer with `private_train()` removed under the
2026-08-03 protocol reset (private gold is test-only) - so the fusion serving
shape has a protocol-legal checkpoint. Bar: fused - score-head >= +0.003 paired
on this checkpoint. Kill: fused <= score head.

Mix: `public_train()` only - RAGTruth EN + 7 translations, HaluEval, PsiloQA,
VitaminC, TabFact - ~686k pairs, 12 domain groups (chance 1/12 ~ 0.083), token
spans where they exist (vitaminc/tabfact masked, H73's rationale). Everything
else byte-faithful to the H102 trainer: mmBERT-base, BCE + CE token head
(W_TOK 1.0) + DANN lambda 0.02 (Ganin ramp), MAX_LEN 512, BATCH 48, LR 1e-5,
SEED 0. RAGBench remains excluded - the arena stays blind.

Evaluation: the three lineage gates (our_gold split, ragtruth_en,
ragtruth_nonen) kept with their recorded bars, plus the `gold_full` read over
ALL 2,752 gold claims (pure test set under the clean-mix protocol, no bar).

Checkpoint saves dann_student.pt with the score head under the "task_head" key
(frozen arena scores it unchanged); the token head rides alongside for the
fusion read (R9-H106_fusion_read.py).

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      uv run python experiments/grounding-semantic/R9-H106_twohead_clean.py
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
CKPT_DIR = HERE.parent.parent / "models" / "R9-H106-twohead-clean"
OUT = HERE / "R9-H106_result.json"

STUDENT = "jhu-clsp/mmBERT-base"  # 307M, under the 400M ceiling, same size as the incumbent
MAX_LEN = 512
BATCH = 48
LR = 1e-5
WARMUP_FRAC, CLIP = 0.1, 1.0
W_TOK = 1.0  # weight of the token-head loss term (H73's value)
SEED = 0

# DANN: the adversarial weight, unchanged from the holder recipe.
LAMBDA_MAX = 0.02
DANN_HIDDEN = 256

# Lineage bars (split-gate era, models then trained in-domain) - kept for
# comparability only; the clean model sits out-of-domain on gold.
BARS = {"gold": (0.7095, 0.76), "ragtruth_en": (0.7039, 0.75), "ragtruth_nonen": (0.6095, 0.66)}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M59 = _mod("m59", "R7-H59_cross_domain_matrix.py")
M60 = _mod("m60", "R7-H60_multilingual_parallel.py")

# spans convention per pair (H73): None -> no token supervision (mask all);
# [] -> supervised, fully grounded (all claim tokens 0); [(s,e),...] -> in-span 1.


def public_train():
    """The holder trainer's full mix (RAGTruth, HaluEval, PsiloQA, VitaminC,
    TabFact), RAGBench excluded, with a domain tag per pair for the
    discriminator and per-pair span supervision where it exists."""
    claims, chunks, ys, tags, spans = [], [], [], [], []

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
        tags.append("ragtruth_en")
        spans.append(sp)

    zt = zipfile.ZipFile(DATA / "dataset-ragtruth-translated.zip")
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        nm = next(
            x for x in zt.namelist() if f"ragtruth-{lg}-" in x and x.endswith("__train.parquet")
        )
        d = pl.read_parquet(io.BytesIO(zt.read(nm)))
        d = d.with_columns((pl.col("labels").list.len() == 0).cast(pl.Float32).alias("label"))
        d = d.filter(pl.col("prompt").str.len_chars() > 50)
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
            tags.append(f"ragtruth_{lg}")
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
        for ev, pos, neg in zip(
            d[ev_col].to_list(), d[pos_col].to_list(), d[neg_col].to_list(), strict=True
        ):
            ev = ev[: M59.CFG.chunk_max_chars]
            claims += [pos, neg]
            chunks += [ev, ev]
            ys += [1.0, 0.0]
            tags += ["halueval", "halueval"]
            spans += [[], [(0, len(neg))]]  # whole-answer span on the hallucinated twin

    zp = zipfile.ZipFile(DATA / "dataset-psiloqa.zip")
    dp = pl.read_parquet(
        io.BytesIO(zp.read(next(x for x in zp.namelist() if x.endswith("__train.parquet"))))
    )
    dp = dp.filter(
        (pl.col("wiki_passage").str.len_chars() > 50) & (pl.col("llm_answer").str.len_chars() > 10)
    ).with_columns((pl.col("labels").list.len() == 0).cast(pl.Float32).alias("label"))
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
        tags.append("psiloqa")
        spans.append([(int(p[0]), int(p[1])) for p in (ls or [])])

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
    ys += dv["y"].to_list()
    tags += ["vitaminc"] * len(dv)
    spans += [None] * len(dv)  # atomic claims: wrong claim, not hallucinated span

    # TabFact - counterfactual statements, same masking rationale as vitaminc.
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
    ys += dt["label"].cast(pl.Float32).to_list()
    tags += ["tabfact"] * len(dt)
    spans += [None] * len(dt)

    return claims, chunks, np.array(ys, dtype="float32"), tags, spans


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


class TwoHeadGroupSet(Dataset):
    def __init__(self, claims, chunks, y, groups, spans, tok):
        self.c, self.k, self.y, self.g, self.s, self.tok = claims, chunks, y, groups, spans, tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.c[i], self.k[i], self.y[i], self.g[i], self.s[i]

    def collate(self, b):
        c, k, y, g, sp = zip(*b, strict=True)
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
        return enc, torch.tensor(y), torch.tensor(g), tok_labels


class GradReverse(torch.autograd.Function):
    """Identity forward, negated-scaled backward - the gradient reversal layer."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class DANNTwoHead(nn.Module):
    """mmBERT trunk, score head, token-span head, and a domain discriminator.

    The score head is the holder recipe's task head (Linear([CLS]) -> 1 logit);
    the token head tags every position (Linear(hidden) -> 2 logits); the domain
    discriminator reads the GRL'd [CLS] exactly as in the holder trainer.
    """

    def __init__(self, base, n_groups, hidden=DANN_HIDDEN):
        super().__init__()
        self.trunk = base
        d = base.config.hidden_size
        self.score_head = nn.Linear(d, 1)
        self.token_head = nn.Linear(d, 2)
        self.domain_head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, n_groups)
        )

    def forward(self, enc, lam):
        h = self.trunk(**enc).last_hidden_state
        cls = h[:, 0]
        return (
            self.score_head(cls).squeeze(-1),
            self.token_head(h),
            self.domain_head(GradReverse.apply(cls, lam)),
        )


@torch.inference_mode()
def score_student(model, tok, claims, chunk_lists):
    """Per-pair fused/score/token probs, max-over-chunks per claim (H73's read)."""
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
        s_logit, t_logit, _ = model(enc, 0.0)
        p_sc[i : i + 64] = torch.sigmoid(s_logit.float()).cpu().numpy()
        halluc = torch.softmax(t_logit.float(), dim=-1)[:, :, 1].cpu()
        halluc[~seq0] = 0.0  # only claim tokens vote
        p_tk[i : i + 64] = (1.0 - halluc.max(dim=1).values).numpy()
    owner = np.array(owner)
    fused = (p_sc + p_tk) / 2.0
    agg = lambda v: np.array([v[owner == i].max() for i in range(len(claims))])
    return agg(fused), agg(p_sc), agg(p_tk)


def evaluate(model, tok):
    """Three lineage gates plus the full-gold read; score head is the guardrail,
    token and fused recorded."""
    res = {}
    sub = _mod("sub", "R8_score_substrate.py")

    claims, chunk_lists, y, _, _ = sub.our_gold()
    f, s, t = score_student(model, tok, claims, chunk_lists)
    res["gold"] = {
        "n": len(y),
        "auc_score_head": round(M59.auc_and_f1(y, s)[0], 4),
        "auc_token_head": round(M59.auc_and_f1(y, t)[0], 4),
        "auc_fused": round(M59.auc_and_f1(y, f)[0], 4),
    }

    cl_f, ck_f, y_f = gold_full()
    f, s, t = score_student(model, tok, cl_f, ck_f)
    res["gold_full"] = {
        "n": len(y_f),
        "auc_score_head": round(M59.auc_and_f1(y_f, s)[0], 4),
        "auc_token_head": round(M59.auc_and_f1(y_f, t)[0], 4),
        "auc_fused": round(M59.auc_and_f1(y_f, f)[0], 4),
    }

    cl, ctx, y = M60.load_english()
    chunks = [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
    f, s, t = score_student(model, tok, cl, chunks)
    res["ragtruth_en"] = {
        "n": len(y),
        "auc_score_head": round(M59.auc_and_f1(y, s)[0], 4),
        "auc_token_head": round(M59.auc_and_f1(y, t)[0], 4),
        "auc_fused": round(M59.auc_and_f1(y, f)[0], 4),
    }

    per_lang_s, per_lang_t = {}, {}
    for lg in ("de", "fr", "es", "it", "pl", "hu", "cn"):
        cl, ctx, y = M60.load_translated(lg)
        _, s, t = score_student(
            model, tok, cl, [M59.top_chunks(c, M59.CFG.semantic_top_k) for c in ctx]
        )
        per_lang_s[lg] = round(M59.auc_and_f1(y, s)[0], 4)
        per_lang_t[lg] = round(M59.auc_and_f1(y, t)[0], 4)
    res["ragtruth_nonen"] = {
        "auc_score_head": round(float(np.mean(list(per_lang_s.values()))), 4),
        "auc_token_head": round(float(np.mean(list(per_lang_t.values()))), 4),
        "per_lang_score": per_lang_s,
        "per_lang_token": per_lang_t,
    }
    return res


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    claims, chunks, y, tags, spans = public_train()
    tag_names = sorted(set(tags))
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    groups = np.array([tag_to_idx[t] for t in tags])
    n_groups = len(tag_names)
    chance = 1.0 / n_groups
    n_tok = sum(1 for s in spans if s is not None)
    n_spanned = sum(1 for s in spans if s)
    print(
        f"train: {len(y)} pairs across {n_groups} domains (chance {chance:.3f}), "
        f"mean target {y.mean():.3f}; token supervision on {n_tok} pairs "
        f"({n_spanned} carry at least one span)  [clean mix - no private pairs]\n",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(STUDENT)
    base = AutoModel.from_pretrained(STUDENT).cuda()
    model = DANNTwoHead(base, n_groups).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"student {STUDENT} + three heads  {n_par:.1f}M params  (ceiling 400M)\n", flush=True)

    ds = TwoHeadGroupSet(claims, chunks, y, groups, spans, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=ds.collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_steps = len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=n_steps, pct_start=WARMUP_FRAC, anneal_strategy="linear"
    )
    score_lossf = nn.BCEWithLogitsLoss()
    token_lossf = nn.CrossEntropyLoss(ignore_index=-100)
    domain_lossf = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    dom_correct, dom_total = 0, 0
    for step, (enc, yy, gg, tl) in enumerate(dl):
        enc = {k: v.cuda() for k, v in enc.items()}
        yy, gg, tl = yy.cuda(), gg.cuda(), tl.cuda()
        # Ganin ramp: lambda grows from 0 to LAMBDA_MAX over training.
        p = step / max(n_steps - 1, 1)
        lam = LAMBDA_MAX * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        s_logit, t_logit, domain_logit = model(enc, lam)
        s_loss = score_lossf(s_logit, yy)
        if (tl != -100).any():
            tok_loss = token_lossf(t_logit.reshape(-1, 2), tl.reshape(-1))
        else:
            tok_loss = torch.zeros((), device=s_logit.device)
        d_loss = domain_lossf(domain_logit, gg)
        loss = s_loss + W_TOK * tok_loss + d_loss  # d_loss lam-scaled inside the GRL

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
                f"  step {step}/{n_steps} score {s_loss.item():.4f} token {tok_loss.item():.4f} "
                f"domain {d_loss.item():.4f} lam {lam:.4f}  domain-acc {acc:.3f} "
                f"(chance {chance:.3f})  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # Score head saved under "task_head" so the frozen arena scores it unchanged;
    # the token head rides alongside for R9-H106_fusion_read.py.
    torch.save(
        {
            "trunk": base.state_dict(),
            "task_head": model.score_head.state_dict(),
            "token_head": model.token_head.state_dict(),
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
    res["w_tok"] = W_TOK
    res["mix"] = "clean (public only, no private pairs)"

    print("=" * 96)
    print("R9-H106 RESULT - clean-mix two-head DANN, in-domain gate (score head vs lineage bars)")
    print("=" * 96)
    print(f"{'corpus':18s} {'score':>9} {'token':>9} {'fused':>9} {'bar':>9} {'':>10}")
    won = 0
    for key, (bar, decisive) in BARS.items():
        r = res[key]
        a = r["auc_score_head"]
        mark = "DECISIVE" if a >= decisive else ("beat" if a > bar else "LOSE")
        won += a > bar
        fused = r.get("auc_fused", float("nan"))
        print(
            f"{key:18s} {a:>9.4f} {r['auc_token_head']:>9.4f} {fused:>9.4f} {bar:>9.4f}  {mark}"
        )
    gf = res["gold_full"]
    print(
        f"{'gold_full':18s} {gf['auc_score_head']:>9.4f} {gf['auc_token_head']:>9.4f} "
        f"{gf['auc_fused']:>9.4f} {'no bar':>9s}  first clean measurement (n={gf['n']})"
    )
    print(f"\n  lineage corpora beaten (score head): {won}/3   params {n_par:.1f}M / 400M")
    print("  blind arena: fused/score/token windowed reads via R9-H106_fusion_read.py")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  results -> {OUT}")


if __name__ == "__main__":
    main()
