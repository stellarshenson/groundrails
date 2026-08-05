"""R10-H111 stage 0 - dropout-dial calibration precursor.

Pre-registered in docs/experiments/semantic-grounding-experiments.md (round 10).
Measures the corruption spectrum (paraphrase / fluent drift / noise) of a
denoising encoder-decoder under inference-time MC dropout, as a function of
dropout p, on ~3k in-register seed statements. Adjudication is external; this
script only reports the composition curves and dumps the eyeball sample.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      uv run python experiments/grounding-semantic/R10-H111_stage0.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import difflib
import json
import pathlib
import random
import re

import numpy as np
import polars as pl
import torch

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
OUT_JSON = HERE / "R10-H111_stage0_result.json"
OUT_EYE = HERE / "R10-H111_eyeball.md"

P_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
N_PER_REGISTER = 1000
SEED = 0
GEN_BATCH = 48
FLUENCY_PCTL = 95  # threshold = this percentile of the p=0.05 NLL distribution
FIDELITY_MIN = 0.75  # mean difflib ratio at p=0 over 20 probes

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEV = "cuda"


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- seeds
def _clean(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def load_seeds():
    seeds = []  # (register, text)
    a = pl.read_parquet(ROOT / "data" / "external" / "datasets" / "R10-H107_pairs.parquet")
    a = a.filter(pl.col("label") == 1)
    for tag, n in (("proc_code", 500), ("proc_gov", 500)):
        rows = (
            a.filter(pl.col("tag") == tag)
            .unique(subset=["claim"])
            .sample(n=n, seed=SEED)["claim"]
            .to_list()
        )
        seeds += [("procedural", _clean(t)) for t in rows]

    b = pl.read_parquet(HERE / "R10-H108_pairs.parquet")
    b = b.filter(pl.col("label") == 1).unique(subset=["claim"])
    rows = b.sample(n=N_PER_REGISTER, seed=SEED)["claim"].to_list()
    seeds += [("quantitative", _clean(t)) for t in rows]

    # hedged-scientific: arXiv abstracts via the CC0 metadata mirror (parquet-
    # native; the classic scientific_papers dataset is a retired script-dataset).
    # No pubmed material anywhere.
    from datasets import load_dataset

    log("streaming arxiv abstracts for scientific seeds ...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    hedge = re.compile(
        r"\b(suggest|may|might|likely|appear|indicate|could|potentially|"
        r"we find|we show|results show|is associated|tend to)\b",
        re.I,
    )
    sci, sci_hedged = [], []
    for i, ex in enumerate(ds):
        if i > 4000 or (len(sci_hedged) >= 500 and len(sci) >= 500):
            break
        abstract = _clean(ex.get("abstract", ""))
        for sent in re.split(r"(?<=[.!?])\s+", abstract):
            w = sent.split()
            if not (12 <= len(w) <= 45) or not sent[0].isupper():
                continue
            if hedge.search(sent):
                if len(sci_hedged) < 500:
                    sci_hedged.append(sent)
            elif len(sci) < 500:
                sci.append(sent)
    seeds += [("scientific", t) for t in (sci_hedged + sci)[:N_PER_REGISTER]]

    # cap length so decoding stays bounded
    seeds = [(r, t[:350]) for r, t in seeds if len(t) >= 40]
    log(f"seeds: {len(seeds)} total "
        f"({sum(1 for r, _ in seeds if r == 'procedural')} procedural, "
        f"{sum(1 for r, _ in seeds if r == 'quantitative')} quantitative, "
        f"{sum(1 for r, _ in seeds if r == 'scientific')} scientific)")
    return seeds


# --------------------------------------------------------------------- generator
def load_mbart():
    from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

    tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50", dtype=torch.float16
    ).to(DEV)
    model.eval()
    return tok, model


def set_dropout(model, p):
    """Set BOTH nn.Dropout modules and HF's float dropout attrs (Bart-family
    applies F.dropout with float attrs `dropout` / `activation_dropout`)."""
    n_mod, n_attr = 0, 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = p
            n_mod += 1
        for attr in ("dropout", "activation_dropout"):
            if hasattr(m, attr) and isinstance(getattr(m, attr), float):
                setattr(m, attr, p)
                n_attr += 1
    return n_mod, n_attr


@torch.no_grad()
def reconstruct(tok, model, texts, p, train_mode):
    """Greedy decode reconstructions at dropout p. train_mode activates dropout."""
    set_dropout(model, p)
    model.train(mode=train_mode)
    outs = []
    bos = tok.lang_code_to_id["en_XX"]
    for i in range(0, len(texts), GEN_BATCH):
        batch = texts[i : i + GEN_BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEV)
        gen = model.generate(
            **enc,
            forced_bos_token_id=bos,
            num_beams=1,
            do_sample=False,
            max_new_tokens=110,
        )
        outs += tok.batch_decode(gen, skip_special_tokens=True)
        if i // GEN_BATCH % 10 == 0:
            log(f"    p={p} batch {i // GEN_BATCH + 1}/{(len(texts) - 1) // GEN_BATCH + 1}")
    model.eval()
    return [_clean(o) for o in outs]


# ----------------------------------------------------------------------- referees
@torch.no_grad()
def gpt2_nll(texts):
    """Length-normalized NLL under gpt2 (EN referee - seeds are EN; recorded
    limitation for any future multilingual use)."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    lm = GPT2LMHeadModel.from_pretrained("gpt2", dtype=torch.float16).to(DEV).eval()
    scores = []
    for i in range(0, len(texts), 64):
        batch = [t if t else "." for t in texts[i : i + 64]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEV)
        out = lm(**enc, labels=enc.input_ids)
        logits = out.logits[:, :-1]
        labels = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].bool()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), reduction="none"
        ).reshape(labels.shape)
        per = (nll * mask).sum(1) / mask.sum(1).clamp(min=1)
        scores += per.tolist()
    del lm
    torch.cuda.empty_cache()
    return scores


@torch.no_grad()
def nli_entail(pairs):
    """P(entailment) for (premise, hypothesis) pairs via mDeBERTa-v3 mnli-xnli."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float16).to(DEV).eval()
    ent_idx = [i for i, l in model.config.id2label.items() if "entail" in l.lower()][0]
    argmax_ent, p_ent = [], []
    for i in range(0, len(pairs), 64):
        batch = pairs[i : i + 64]
        enc = tok(
            [a for a, _ in batch],
            [b for _, b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEV)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)
        argmax_ent += (probs.argmax(-1) == ent_idx).tolist()
        p_ent += probs[:, ent_idx].tolist()
        if i // 64 % 40 == 0:
            log(f"    nli batch {i // 64 + 1}/{(len(pairs) - 1) // 64 + 1}")
    del model
    torch.cuda.empty_cache()
    return argmax_ent, p_ent


# --------------------------------------------------------------------------- main
def main():
    seeds = load_seeds()
    registers = [r for r, _ in seeds]
    texts = [t for _, t in seeds]

    tok, model = load_mbart()

    # fidelity gate: p=0, dropout OFF, 20 probes - near-verbatim required
    probe = texts[:: max(1, len(texts) // 20)][:20]
    rec0 = reconstruct(tok, model, probe, p=0.0, train_mode=False)
    fid = float(np.mean([difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() for a, b in zip(probe, rec0)]))
    log(f"FIDELITY mbart-large-50 @p=0: mean ratio {fid:.3f} (min {FIDELITY_MIN})")
    for a, b in list(zip(probe, rec0))[:3]:
        log(f"  seed: {a[:110]}\n  rec : {b[:110]}")
    if fid < FIDELITY_MIN:
        log("FIDELITY FAIL - mbart cannot identity-reconstruct; STOPPING per directive "
            "(mt5 identity-finetune fallback requires a separate decision)")
        OUT_JSON.write_text(json.dumps({"fidelity_fail": True, "fidelity": fid}, indent=1))
        return

    # sweep
    recs = {}  # p -> list of reconstructions
    for p in P_GRID:
        log(f"sweep p={p} ...")
        torch.manual_seed(SEED + int(p * 100))
        recs[p] = reconstruct(tok, model, texts, p=p, train_mode=True)
    del model
    torch.cuda.empty_cache()

    # fluency referee
    log("fluency referee (gpt2) ...")
    all_nll = {p: gpt2_nll(recs[p]) for p in P_GRID}
    thresh = float(np.percentile(all_nll[P_GRID[0]], FLUENCY_PCTL))
    log(f"fluency threshold (p{FLUENCY_PCTL} of p={P_GRID[0]} NLL): {thresh:.3f}")

    # NLI referee, both directions, per p
    comp = {}
    per_reg = {}
    detail = {}
    for p in P_GRID:
        log(f"nli referee p={p} ...")
        fwd_am, fwd_pe = nli_entail(list(zip(texts, recs[p])))
        bwd_am, bwd_pe = nli_entail(list(zip(recs[p], texts)))
        cls, minent = [], []
        for i in range(len(texts)):
            fluent = all_nll[p][i] <= thresh
            exact = recs[p][i].lower() == texts[i].lower()
            para = (fwd_am[i] and bwd_am[i]) or exact
            if para and fluent:
                cls.append("paraphrase")
            elif fluent:
                cls.append("drift")
            else:
                cls.append("noise")
            minent.append(min(fwd_pe[i], bwd_pe[i]))
        n = len(cls)
        comp[p] = {
            "paraphrase": round(cls.count("paraphrase") / n, 4),
            "drift": round(cls.count("drift") / n, 4),
            "noise": round(cls.count("noise") / n, 4),
            "exact_match": round(np.mean([recs[p][i].lower() == texts[i].lower() for i in range(n)]), 4),
            "n": n,
        }
        for reg in ("procedural", "quantitative", "scientific"):
            idx = [i for i in range(n) if registers[i] == reg]
            per_reg.setdefault(reg, {})[p] = {
                "paraphrase": round(sum(cls[i] == "paraphrase" for i in idx) / len(idx), 4),
                "drift": round(sum(cls[i] == "drift" for i in idx) / len(idx), 4),
                "noise": round(sum(cls[i] == "noise" for i in idx) / len(idx), 4),
            }
        detail[p] = {"cls": cls, "minent": minent}
        log(f"  p={p}: {comp[p]}")

    best_p = max(P_GRID, key=lambda p: comp[p]["drift"])

    # eyeball dump at best_p
    cls = detail[best_p]["cls"]
    minent = detail[best_p]["minent"]
    drift_idx = [i for i in range(len(texts)) if cls[i] == "drift"]
    para_idx = sorted(
        [i for i in range(len(texts)) if cls[i] == "paraphrase"], key=lambda i: minent[i]
    )
    rng = random.Random(SEED)
    pick_drift = rng.sample(drift_idx, min(50, len(drift_idx)))
    pick_border = para_idx[:25]
    lines = [
        "# R10-H111 stage 0 - eyeball sample",
        "",
        f"Model facebook/mbart-large-50, best_p {best_p}, fluency threshold {thresh:.3f} "
        f"(p{FLUENCY_PCTL} of p={P_GRID[0]} NLL). Adjudication bar: < 1 in 10 of the",
        "ADMITTED DRIFT pairs below is actually a meaning-preserving paraphrase.",
        "",
        f"## Admitted drift at p={best_p} ({len(pick_drift)} random)",
        "",
    ]
    for k, i in enumerate(pick_drift):
        lines += [
            f"**D{k + 1}** [{registers[i]}] min-entail {minent[i]:.2f}",
            f"- seed: {texts[i]}",
            f"- rec : {recs[best_p][i]}",
            "",
        ]
    lines += [f"## Borderline paraphrase (25 lowest min-entailment among admitted paraphrases)", ""]
    for k, i in enumerate(pick_border):
        lines += [
            f"**B{k + 1}** [{registers[i]}] min-entail {minent[i]:.2f}",
            f"- seed: {texts[i]}",
            f"- rec : {recs[best_p][i]}",
            "",
        ]
    OUT_EYE.write_text("\n".join(lines))

    result = {
        "model_used": "facebook/mbart-large-50",
        "fidelity_at_p0": fid,
        "dropout_mechanism": "model.train() + set p on every nn.Dropout module AND every float "
        "`dropout`/`activation_dropout` attr (Bart-family F.dropout paths); greedy decode, "
        "noise source is dropout only",
        "fluency_referee": "gpt2 length-normalized NLL, threshold p95 of the p=0.05 distribution "
        "(EN-only referee - seeds are EN; multilingual limitation recorded)",
        "nli_referee": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, both directions, argmax entailment",
        "scientific_seed_source": "gfissore/arxiv-abstracts-2021 (streaming; arXiv metadata "
        "mirror, CC0; no pubmed material)",
        "per_p": comp,
        "per_register": per_reg,
        "best_p": best_p,
        "drift_yield_at_best_p": comp[best_p]["drift"],
        "examples_path": str(OUT_EYE),
    }
    OUT_JSON.write_text(json.dumps(result, indent=1))
    log(json.dumps({k: v for k, v in result.items() if k in ("per_p", "best_p", "drift_yield_at_best_p")}, indent=1))
    log(f"results -> {OUT_JSON}")
    log("=== R10-H111 STAGE0 DONE ===")


if __name__ == "__main__":
    main()
